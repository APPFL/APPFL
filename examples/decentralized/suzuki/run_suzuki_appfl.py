"""Reproduce the ADKO ``scientific_discovery`` Suzuki study on APPFL's decentralized runtime.

    python examples/decentralized/suzuki/run_suzuki_appfl.py \
        --config ../adko/scientific_discovery/experiments/main_iid_llmoff.json \
        --warmup-dir ../adko/scientific_discovery/results/warmup \
        --out-dir results/appfl_main

Reads the reference's own experiment JSON and writes results in the reference's own result
schema, so ``scientific_discovery/analytics/*.py`` consumes both without modification and the
two can be diffed directly.

This is the LM-free arm (``use_llm: false``). That is the interesting one for a port check:
it isolates whether token-based collaboration reproduces, with no language model in the loop
to absorb a discrepancy.

Three settings are not free parameters -- they are what make this the *same* experiment rather
than a similar one, and each is a place the two implementations would otherwise silently
diverge. See ``ADKOAgent``'s docstring for the full argument:

* ``include_own_tokens=False``  -- the reference drops an agent's own token on receipt
  (``run_suzuki.py:1358``) but leaves ``pi`` assuming it arrived; see ``ADKOAgent``.
* ``embedding_privatizer``      -- ``p_noise`` applies to the token embedding only.
* ``warmup_points``             -- replay the shared warmup bank, not a fresh RNG stream.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from appfl.decentralized import InProcessExchange, build_topology, run_federation
from appfl.decentralized.algorithm.adko import (
    ADKOAgent,
    ADKOMeter,
    FidelityAwarePruner,
    ReasoningWeights,
    build_baseline,
)

from space import (  # noqa: E402  -- sys.path is set above
    D,
    N_OPTS_PER_DIM,
    SuzukiSpace,
    build_lookup_table,
    decode_int_to_smiles,
    make_evaluator,
)
from surrogate import CategoricalGPSurrogate  # noqa: E402

N_AGENTS = 4
N_ROUNDS = 200
WARMUP_ROUNDS = 5
SEED_BASE = 42  # `"seeds": 40` in a config means seeds 42..81 (`run_suzuki.py:363`)


# ---------------------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------------------


def _scalar(value: Any) -> Any:
    """Reference configs allow a 1-element list anywhere a scalar is meant."""
    return value[0] if isinstance(value, list) else value


def load_config(path: Path) -> Dict[str, Any]:
    cfg = json.loads(path.read_text())
    if bool(cfg.get("use_llm")):
        raise SystemExit(
            f"{path}: use_llm is true. This driver reproduces the LM-free arm only; "
            f"attach a LanguageModel to ADKOAgent to go further."
        )
    methods = [str(m) for m in (cfg["methods"] if isinstance(cfg["methods"], list) else [cfg["methods"]])]
    unsupported = [m for m in methods if m not in ("ADKO", "INDEP")]
    if unsupported:
        raise SystemExit(
            f"{path}: methods {unsupported} have no counterpart in appfl.decentralized "
            f"(they are server-coordinated baselines). Supported: ADKO, INDEP."
        )
    tau = float(_scalar(cfg["tau"]))
    return {
        "methods": methods,
        "iid_modes": [bool(v) for v in (cfg["iid_mode"] if isinstance(cfg["iid_mode"], list) else [cfg["iid_mode"]])],
        "seeds": [SEED_BASE + i for i in range(int(cfg["seeds"]))],
        "beta": float(_scalar(cfg["beta"])),
        "lam": float(_scalar(cfg["lam"])),
        "gamma": float(_scalar(cfg["gamma"])),
        "token_budget": int(_scalar(cfg["token_budget"])),
        "alpha_tau": float(_scalar(cfg["alpha_tau"])),
        "p_noise": float(_scalar(cfg["p_noise"])),
        "similarity_bw": float(_scalar(cfg["similarity_bw"])),
        "tau": tau,
        # `ACTIVE_Y_SCALE = max(tau, 100 - tau)` (`run_suzuki.py:178`). tau=50 -> 50.
        "y_scale": float(max(tau, 100.0 - tau)),
        "total_proposals": cfg["total_proposals"],
        "parallel": int(cfg.get("parallel", 1)),
    }


def load_warmup(
    warmup_dir: Optional[Path], seed: int, iid_mode: bool
) -> Optional[List[List[Sequence[int]]]]:
    """Shared warmup bank -> ``warmup_points[agent][round]``.

    Returns ``None`` when no bank is supplied, in which case each agent draws its own warmup
    at random and only a distributional comparison against the reference is meaningful.
    """
    if warmup_dir is None:
        return None
    path = warmup_dir / f"{'IID' if iid_mode else 'HET'}_seed{seed}.json"
    if not path.exists():
        raise SystemExit(f"warmup bank not found: {path}")
    payload = json.loads(path.read_text())
    if bool(payload["iid_mode"]) != bool(iid_mode) or int(payload["seed"]) != seed:
        raise SystemExit(f"{path}: bank does not match seed={seed} iid={iid_mode}")

    points: List[List[Optional[Sequence[int]]]] = [
        [None] * int(payload["warmup_rounds"]) for _ in range(int(payload["n_agents"]))
    ]
    for rec in payload["observations"]:
        points[int(rec["agent"])][int(rec["round"])] = tuple(
            int(v) for v in rec["theta_int"]
        )
    for a, rounds in enumerate(points):
        if any(p is None for p in rounds):
            raise SystemExit(f"{path}: incomplete warmup for agent {a}")
    return points  # type: ignore[return-value]


# ---------------------------------------------------------------------------------------
# one run
# ---------------------------------------------------------------------------------------


class TokenEmbeddingPrivatizer:
    """Reference-compatible token-location noise with a resettable RNG."""

    def __init__(self, p_noise: float, seed: int):
        self.p_noise = float(p_noise)
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def __call__(self, point: Sequence[float]) -> List[float]:
        noisy = [float(v) for v in point]
        if self.p_noise <= 0.0:
            return noisy
        for dim, n_opts in enumerate(N_OPTS_PER_DIM):
            if n_opts <= 1:
                continue
            if float(self.rng.random()) < self.p_noise:
                true_val = int(noisy[dim])
                alt = int(self.rng.integers(0, n_opts - 1))
                if alt >= true_val:
                    alt += 1
                noisy[dim] = float(alt)
        return noisy


def post_warmup_agent_seed(seed: int, agent_index: int) -> int:
    """Reference ``_post_warmup_agent_seed``: seed 42 -> agent seeds 42, 43, 44, 45."""
    return int(seed) + int(agent_index)


def run_one(
    seed: int,
    iid_mode: bool,
    method: str,
    cfg: Dict[str, Any],
    lookup: Dict[Any, float],
    warmup_dir: Optional[Path],
) -> Dict[str, Any]:
    """One (seed, heterogeneity, method) run. Mirrors ``run_adko.run_adko``."""
    import torch

    started = time.time()
    torch.manual_seed(seed)
    agent_ids = [f"agent-{i}" for i in range(N_AGENTS)]
    topology = build_topology("fully_connected", agent_ids)
    assert topology.fiedler_value() > 1e-9, "disconnected graph; ADKO's guarantees do not apply"

    evaluate = make_evaluator(lookup)
    warmup = load_warmup(warmup_dir, seed, iid_mode)

    # The reference derives a per-agent RNG from the run seed (`run_adko.py:78`). Token noise
    # is the only consumer of it here, since the candidate pool is the full grid.
    root_rng = np.random.default_rng(seed)
    agent_seeds = [int(root_rng.integers(0, 2**31)) for _ in range(N_AGENTS)]

    weights = ReasoningWeights(
        beta=cfg["beta"],
        lam=cfg["lam"],
        gamma=cfg["gamma"],
        sigma_s=cfg["similarity_bw"],
        metric="hamming",
        kernel="two_sigma_sq",
        weight_by_fidelity=True,
        peer_normalization="mixing_weight",
    )

    meter = ADKOMeter()
    exchange = InProcessExchange(topology, meter=meter)

    agents: List[ADKOAgent] = []
    privatizers: List[TokenEmbeddingPrivatizer] = []
    for i, agent_id in enumerate(agent_ids):
        privatizer = TokenEmbeddingPrivatizer(cfg["p_noise"], agent_seeds[i])
        privatizers.append(privatizer)
        agents.append(
            ADKOAgent(
                agent_id=agent_id,
                surrogate=CategoricalGPSurrogate(),
                space=SuzukiSpace(i, iid_mode, seed=agent_seeds[i]),
                evaluator=evaluate,
                mixing_weight=partial(topology.uniform_weight, agent_id),
                baseline=build_baseline(
                    "fixed", threshold=cfg["tau"], scale=cfg["y_scale"]
                ),
                language_model=None,
                pruner=FidelityAwarePruner(alpha_tau=cfg["alpha_tau"]),
                weights=weights,
                token_budget=cfg["token_budget"],
                alpha_tau=cfg["alpha_tau"],
                emit_insight=False,
                objective="maximize",
                warmup_rounds=WARMUP_ROUNDS,
                total_proposals=cfg["total_proposals"],
                seed=agent_seeds[i],
                # -- the three replication-critical settings, see module docstring --
                include_own_tokens=False,
                embedding_privatizer=privatizer,
                warmup_points=(warmup[i] if warmup is not None else None),
            )
        )

    # INDEP is the ADKO loop with broadcast disabled (`run_adko.py:7`): no tokens reach a
    # peer, token memory stays empty, and Eq. (1) collapses to per-agent GP-UCB.
    if method == "INDEP":
        exchange.publish = lambda agent_id, tokens: None  # type: ignore[assignment]

    best_int = np.full((N_ROUNDS, N_AGENTS), -np.inf)
    system_best = np.zeros(N_ROUNDS)
    mean_eta = np.ones((N_ROUNDS, N_AGENTS))
    steps: List[Dict[str, Any]] = []

    def on_round_end(round_idx: int, round_agents: Sequence[ADKOAgent]) -> None:
        for i, agent in enumerate(round_agents):
            point = agent._points[-1]
            y = float(agent._observations[-1])
            prev = best_int[round_idx - 1, i] if round_idx > 0 else -np.inf
            best_int[round_idx, i] = max(prev, y)
            mean_eta[round_idx, i] = agent.mean_token_fidelity()

            trace = agent.last_decision or {}
            steps.append(
                {
                    "round": round_idx,
                    "agent": i,
                    "theta_int": [int(v) for v in point],
                    "theta_phys": decode_int_to_smiles(point),
                    "y_internal": y,
                    "y_raw": y,
                    "best_so_far_internal": float(best_int[round_idx, i]),
                    "best_so_far_raw": float(best_int[round_idx, i]),
                    "score_mu": trace.get("score_mu", 0.0),
                    "score_beta_sigma": trace.get("score_beta_sigma", 0.0),
                    "score_lam_G": trace.get("score_lam_G", 0.0),
                    "score_gamma_Lambda": trace.get("score_gamma_Lambda", 0.0),
                    "score_llm": 0.0,
                    "chose_by": (
                        "shared_warmup"
                        if (warmup is not None and round_idx < WARMUP_ROUNDS)
                        else "random_warmup"
                        if not trace
                        else "reasoning_full_grid"
                    ),
                    "n_tokens_in_memory": len(agent.token_memory),
                    "mean_token_eta": agent.mean_token_fidelity(),
                    "n_candidates_scored": trace.get("n_candidates_scored", 0),
                }
            )
        system_best[round_idx] = float(best_int[round_idx, :].max())
        if warmup is not None and round_idx == WARMUP_ROUNDS - 1:
            for i, privatizer in enumerate(privatizers):
                privatizer.reset(post_warmup_agent_seed(seed, i))

    run_federation(agents, exchange, N_ROUNDS, meter=meter, on_round_end=on_round_end)

    return {
        "method": method,
        "heterogeneity": "iid" if iid_mode else "het",
        "seed": seed,
        "config": {
            "n_agents": N_AGENTS,
            "n_rounds": N_ROUNDS,
            "warmup_rounds": WARMUP_ROUNDS,
            "beta": cfg["beta"],
            "lam": cfg["lam"],
            "gamma": cfg["gamma"],
            "token_budget": cfg["token_budget"],
            "alpha_tau": cfg["alpha_tau"],
            "p_noise": cfg["p_noise"],
            "similarity_bandwidth": cfg["similarity_bw"],
            "tau": cfg["tau"],
            "y_scale": cfg["y_scale"],
            "graph_type": "complete",
            "heterogeneity": "iid" if iid_mode else "het",
            "dataset": "suzuki_edbo",
            "gp_backend": "botorch_categorical_single_task_gp",
            "kernel": "ScaleKernel(CategoricalKernel(ARD))",
            "input_space": "integer_categories",
            "candidate_strategy": "full_unobserved_cartesian_product",
            "total_proposals": cfg["total_proposals"],
            "use_llm": False,
            "implementation": "appfl.decentralized",
            "include_own_tokens": False,
            "shared_warmup": warmup is not None,
        },
        "steps": steps,
        "best_int": best_int.tolist(),
        "system_best": system_best.tolist(),
        "mean_token_eta": mean_eta.tolist(),
        "graph_type": "complete",
        "fiedler": topology.fiedler_value(),
        "final_system_best": float(system_best[-1]),
        "wall_seconds": time.time() - started,
    }


# ---------------------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------------------


def _job(spec, cfg, warmup_dir, out_dir):
    seed, iid_mode, method = spec
    tag = f"{'IID' if iid_mode else 'HET'}-{method}_seed{seed}"
    path = out_dir / f"{tag}.json"
    if path.exists():
        return f"[skip] {tag}"
    result = run_one(seed, iid_mode, method, cfg, build_lookup_table(), warmup_dir)
    path.write_text(json.dumps(result))
    return (
        f"[done] {tag}  final={result['final_system_best']:.2f}  "
        f"{result['wall_seconds']:.0f}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="a scientific_discovery/experiments/*.json config")
    parser.add_argument("--warmup-dir", type=Path, default=None,
                        help="scientific_discovery/results/warmup; strongly recommended -- "
                             "without it the arms start from different data")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=None,
                        help="run only the first N seeds (for the Stage-1 check)")
    parser.add_argument("--parallel", type=int, default=None,
                        help="worker processes; defaults to the config's value")
    parser.add_argument("--p-noise", type=float, default=None,
                        help="override p_noise; set 0 for the deterministic Stage-1 check")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.p_noise is not None:
        cfg["p_noise"] = args.p_noise
    if args.seeds is not None:
        cfg["seeds"] = cfg["seeds"][: args.seeds]
    n_workers = args.parallel if args.parallel is not None else cfg["parallel"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        (seed, iid, method)
        for method in cfg["methods"]
        for iid in cfg["iid_modes"]
        for seed in cfg["seeds"]
    ]
    print(f"[run ] {len(specs)} runs -> {args.out_dir}  (parallel={n_workers})", flush=True)

    job = partial(_job, cfg=cfg, warmup_dir=args.warmup_dir, out_dir=args.out_dir)
    if n_workers <= 1:
        for spec in specs:
            print(job(spec), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for line in pool.map(job, specs):
                print(line, flush=True)


if __name__ == "__main__":
    main()
