"""One arm, one command. ``python -m appfl.decentralized.algorithm.adko_h.run --help``

Every arm of the screen is this module with different flags, which is deliberate: an arm that
needs its own script is an arm whose difference is not written down anywhere. The flags are
the method's knobs and nothing else -- the problem, the graph, the planner, and the three
switches the ablations turn (claims, territories, the novelty bonus).

    # plain ADKO v2 over a sampled pool, 8 seeds
    python -m ...adko_h.run --problem pmo --oracle zaleplon_mpo --arm adko --seeds 42-49

    # ADKO-H against a served model
    python -m ...adko_h.run --problem pmo --oracle zaleplon_mpo --arm adkoh \\
        --planner vllm --llm-url http://$HOST:8000/v1 --llm-model Qwen/Qwen3.8-27B-FP8

    # ADKO-H through the AmSC gateway: costs money, so the model and the cap are explicit
    MAG_LEDGER=$ADKO_DATA/mag_ledger.txt MAG_BUDGET=20 \\
    python -m ...adko_h.run --problem pmo --oracle zaleplon_mpo --arm adkoh \\
        --planner mag --llm-model claude-sonnet-5-5-medium@amazon
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from appfl.decentralized.algorithm.adko_h.agent import AgentConfig
from appfl.decentralized.algorithm.adko_h.llm import build_planner
from appfl.decentralized.algorithm.adko_h.loop import RunConfig, run
from appfl.decentralized.algorithm.adko_h.surrogate import GP

#: The arms the screen compares. Each is a configuration, not a code path.
ARMS: Dict[str, Dict[str, Any]] = {
    # no language model at all
    "adko": {"planner": "none"},
    "adko-div": {"planner": "none", "novelty_weight": 1.0},
    # campaigns, no language model: does the structure alone do the work?
    "random-campaign": {"planner": "random"},
    # campaigns planned by a model
    "adkoh": {"planner": "llm"},
    "adkoh-noclaims": {"planner": "llm", "claims_on": False},
    "adkoh-noterritory": {"planner": "llm", "territory": False},
    "adkoh-nofootprint": {"planner": "llm", "share_descriptor": False, "share_family": False},
}


def parse(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="adko_h.run", description=__doc__.splitlines()[0])
    g = p.add_argument_group("problem")
    g.add_argument("--problem", default="synthetic", choices=("synthetic", "pmo"))
    g.add_argument("--oracle", default="albuterol_similarity", help="PMO task name")
    g.add_argument("--pool", default=None, help="molecule pool .smi (default $ADKO_PMO_POOL)")
    g.add_argument("--pool-limit", type=int, default=None, help="use only the first N molecules")
    g.add_argument("--embed-dim", type=int, default=32)

    g = p.add_argument_group("run")
    g.add_argument("--arm", default="adko", choices=sorted(ARMS))
    g.add_argument("--rounds", type=int, default=50)
    g.add_argument("--agents", type=int, default=10)
    g.add_argument("--seeds", default="0", help="e.g. 0, 42-49, or 1,2,5")
    g.add_argument("--topology", default="fully_connected")
    g.add_argument("--degree", type=int, default=4, help="k, for ring and random_regular")
    g.add_argument("--pool-size", type=int, default=256, help="candidates per agent-round")
    g.add_argument("--warmup", type=int, default=3)
    g.add_argument("--start-points", type=int, default=16)
    g.add_argument("--out", default=None, help="results root; nothing is written without it")
    g.add_argument("--label", default=None, help="overrides the arm name in the record")
    g.add_argument("--rule", default=None,
                   help="fix the rule for a no-planner arm (adko, indep, exploit, explore, "
                        "novelty): measures how much choosing a rule can be worth")
    g.add_argument("--verbose", action="store_true")

    g = p.add_argument_group("planner")
    g.add_argument("--planner", default=None, choices=("none", "random", "mock", "vllm", "mag"),
                   help="overrides the arm's default; an LLM arm needs mock, vllm or mag")
    g.add_argument("--llm-url", default=os.environ.get("ADKO_LLM_BASE_URL"))
    g.add_argument("--llm-model", default=None,
                   help="vllm: default Qwen/Qwen3.8-27B-FP8. mag: required, e.g. "
                        "claude-sonnet-5-5-medium@amazon (effort is part of the id)")
    g.add_argument("--llm-thinking", default="low", choices=("low", "medium", "high", "off"),
                   help="vllm only")
    g.add_argument("--llm-temperature", type=float, default=None,
                   help="default 0.2 on vllm, unset on mag (thinking models fix it)")
    g.add_argument("--llm-max-tokens", type=int, default=None,
                   help="default 4096 on vllm, 16000 on mag (room for the model's reasoning)")
    g.add_argument("--no-llm-cache", action="store_true")
    g.add_argument("--campaign-min", type=int, default=2, help="shortest campaign")
    g.add_argument("--campaign-max", type=int, default=10,
                   help="longest campaign; one campaign is one model call, so this sets the "
                        "token cost of a run")

    g = p.add_argument_group("endpoints")
    g.add_argument("--circle-threshold", type=float, default=0.75)
    g.add_argument("--hit-threshold", type=float, default=0.5)
    return p.parse_args(argv)


def seeds_of(spec: str) -> List[int]:
    out: List[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part[1:]:
            a, b = part.split("-", 1)
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def make_problem(args: argparse.Namespace, seed: int):
    if args.problem == "pmo":
        from appfl.decentralized.algorithm.adko_h.problems.pmo import PMO

        return PMO(
            args.oracle,
            pool_path=args.pool,
            limit=args.pool_limit,
            embed_dim=args.embed_dim,
            seed=0,  # the embedding is frozen across seeds: every arm sees one space
            verbose=args.verbose,
        )
    from appfl.decentralized.algorithm.adko_h.problems.synthetic import Synthetic

    return Synthetic(seed=seed)


def main(argv=None) -> int:
    args = parse(argv)
    arm = dict(ARMS[args.arm])
    planner_kind = args.planner or arm.pop("planner")
    arm.pop("planner", None)
    label = args.label or (f"{args.arm}-{args.rule}" if args.rule else args.arm)

    topo_kwargs: Dict[str, Any] = {}
    if args.topology in ("ring", "random_regular"):
        topo_kwargs["k"] = args.degree
    if args.topology in ("random_regular", "random_geometric"):
        topo_kwargs["seed"] = 0  # the graph is held fixed across the method's seeds

    if planner_kind in ("mock", "vllm", "mag"):
        if args.llm_model is None:
            if planner_kind == "mag":
                raise SystemExit("--planner mag needs --llm-model: MAG bills per call, so the "
                                 "model is a decision, not a default")
            args.llm_model = "Qwen/Qwen3.8-27B-FP8"
        if args.llm_max_tokens is None:
            args.llm_max_tokens = 16000 if planner_kind == "mag" else 4096
        if args.llm_temperature is None and planner_kind != "mag":
            args.llm_temperature = 0.2
        if planner_kind == "mag" and not os.environ.get("MAG_BUDGET"):
            raise SystemExit("--planner mag needs MAG_LEDGER and MAG_BUDGET set: a run with "
                             "no spending cap is not something to start by accident")

    for seed in seeds_of(args.seeds):
        problem = make_problem(args, seed)
        out = (
            str(Path(args.out) / f"{problem.problem_id.replace(':', '-')}" / label / f"seed{seed}")
            if args.out
            else None
        )
        planner = None
        if planner_kind in ("mock", "vllm", "mag"):
            planner = build_planner(
                problem.region_space,
                planner_kind,
                base_url=args.llm_url,
                model=args.llm_model,
                run_dir=out,
                seed=seed,
                temperature=args.llm_temperature,
                thinking=None if args.llm_thinking == "off" else args.llm_thinking,
                max_tokens=args.llm_max_tokens,
                use_cache=not args.no_llm_cache,
                budget_range=(args.campaign_min, args.campaign_max),
            )
        cfg = RunConfig(
            rounds=args.rounds,
            agents=args.agents,
            seed=seed,
            topology=args.topology,
            topology_kwargs=topo_kwargs,
            planner="llm" if planner_kind in ("mock", "vllm", "mag") else planner_kind,
            label=label,
            start_points=args.start_points,
            circle_threshold=args.circle_threshold,
            hit_threshold=args.hit_threshold,
            agent_cfg=AgentConfig(
                warmup_rounds=args.warmup,
                pool_size=args.pool_size,
                budget_range=(args.campaign_min, args.campaign_max),
                force_rule=args.rule,
                **arm,
            ),
        )
        record = run(problem, GP, cfg, planner=planner, out=out, verbose=args.verbose)
        if planner is not None:
            planner.close()
        e, c = record["endpoints"], record["campaigns"]
        print(
            f"{label:18s} seed={seed} families={e['distinct_families']:4d} "
            f"hits={e['diverse_hits']:3d} circles={e['n_circles']:4d} "
            f"hull={e['hull_volume']:8.3f} auc={e['auc_top10']:.4f} best={e['best']:.4f} "
            f"campaigns={c['n']}/{c['passed']} brier={_f(c['calibration']['brier'])} "
            f"bits={record['bits_per_token']:.0f} {record['seconds']}s"
            + (f" llm=${record['planner'].get('cost_usd', 0):.4f}/{int(record['planner'].get('designs', 0))} designs"
               if record.get("planner", {}).get("cost_usd") else ""),
            flush=True,
        )
        if out:
            print(f"  -> {out}", flush=True)
    return 0


def _f(v) -> str:
    return "n/a" if v is None else f"{v:.3f}"


if __name__ == "__main__":
    sys.exit(main())
