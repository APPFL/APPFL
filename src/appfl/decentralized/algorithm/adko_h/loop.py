"""The round driver: a library entry point, not an APPFL launcher.

ADKO v2 runs under APPFL's serial and MPI launchers, where one round is ``observe`` then
``act`` per node and the evaluation happens inside ``act``. ADKO-H cannot use that shape for
the problems it targets: a round's picks have to reach the GPUs together, and every agent that
needs a campaign has to be planned in **one** batched language-model call, or a single served
27B model becomes the bottleneck at ten agents and an impossibility at a thousand.

So the loop is explicit here and the phases are separated -- plan, propose, evaluate, complete,
exchange -- which also makes this callable as a library from another harness. That is how an expensive
simulator should use it: its own harness drives the rounds and calls these classes, with
APPFL's topologies and budget doing what they already do.

Exchange respects the graph: an agent observes only its in-neighbours' tokens, one per
neighbour per round, which is ADKO's communication constraint unchanged from v1.

Not implemented: mid-run resume. A crashed run leaves ``rounds.jsonl`` behind for inspection
and is re-run from the start; on an expensive problem the evaluator is expected to cache, which
is where a resume belongs anyway.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from appfl.decentralized.algorithm.adko_h import metrics as hmetrics
from appfl.decentralized.algorithm.adko_h.agent import ADKOHAgent, AgentConfig
from appfl.decentralized.algorithm.adko_h.campaign import BUDGET_RANGE, Test, random_design
from appfl.decentralized.algorithm.adko_h.problem import Pick, Problem
from appfl.decentralized.topology import build_topology


@dataclass
class RunConfig:
    """One arm of a comparison.

    Args:
        rounds: synchronous rounds; the budget is ``rounds * agents`` experiments.
        agents: how many labs.
        seed: the run seed. Agents derive their streams from ``(seed, index)``.
        topology: a name from APPFL's topology registry.
        topology_kwargs: that topology's arguments, e.g. ``{"k": 4, "seed": 0}``.
        agent_cfg: the method's configuration, shared by every agent.
        planner: ``"none"`` (plain ADKO over a pool), ``"random"`` (the campaign control),
            ``"replay"`` (designs replayed from ``replay_from``) or ``"llm"`` (a planner object
            must be passed to :func:`run`).
        start_points: shared initial experiments, loaded into every agent and not charged to
            the budget -- the reference comparisons do this so arms are paired by seed.
        circle_threshold: distance for the ``n_circles`` endpoint.
        hit_threshold: score above which an outcome counts as a hit for ``diverse_hits``.
        random_center: where the random planner centres its boxes: ``"uniform"`` anywhere in
            the unit cube, or ``"pool"`` on a randomly drawn candidate, so boxes sit where
            candidates exist.
        random_widths: the random planner's half-width per region coordinate (normalized
            units); ``None`` draws one random half-width per box.
        replay_from: a JSON list of designs (``bounds``, ``picks``, and optionally ``rule``,
            ``budget``, ``test``) for the replay planner.
    """

    rounds: int = 50
    agents: int = 10
    seed: int = 0
    topology: str = "fully_connected"
    topology_kwargs: Dict[str, Any] = field(default_factory=dict)
    agent_cfg: AgentConfig = field(default_factory=AgentConfig)
    planner: str = "none"
    label: str = ""
    start_points: int = 0
    circle_threshold: float = 0.75
    hit_threshold: float = 0.8
    random_center: str = "uniform"
    random_widths: Optional[Tuple[float, ...]] = None
    replay_from: Optional[str] = None


class RandomPlanner:
    """The control: the same campaign lifecycle with designs drawn at random.

    The arm that says whether ADKO-H's gain is the planning or merely the structure -- regions,
    budgets, claims and verdicts all still happen, just without anything thinking about them.

    Args:
        anchor: draws a unit-cube point to centre each box on (e.g. a random candidate's
            normalized coordinates); ``None`` centres boxes uniformly in the cube.
        widths: one half-width per region coordinate; ``None`` draws one per box.
    """

    def __init__(self, space, seed: int = 0, budget_range=None, anchor=None, widths=None):
        self.space = space
        self.budget_range = tuple(budget_range) if budget_range else BUDGET_RANGE
        self.rng = np.random.default_rng([seed, 7])
        self.anchor = anchor
        self.widths = None if widths is None else tuple(float(w) for w in widths)
        self.totals: Dict[str, float] = {"calls": 0, "designs": 0}

    def plan(self, contexts: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.totals["calls"] += 1
        self.totals["designs"] += len(contexts)
        return [
            random_design(
                self.space,
                self.rng,
                budget_range=self.budget_range,
                center=None if self.anchor is None else self.anchor(self.rng),
                widths=self.widths,
            )
            for _ in contexts
        ]

    def record_outcome(self, campaign, verdict) -> None:
        return None

    def close(self) -> None:
        return None


class ReplayPlanner:
    """Another arm's designs, replayed in random order: its regions without its reasons.

    Each request draws one design from the list, so the replayed regions keep their
    distribution and lose their timing and their context -- whose history they answered, and
    when. Comparing this with the arm that produced them separates what a planner's regions
    are worth on their own (a prior a single call could capture) from what it gains by
    choosing them for this agent at this moment.
    """

    def __init__(self, space, designs: Sequence[Dict[str, Any]], seed: int = 0, budget_range=None):
        if not designs:
            raise ValueError("a replay planner needs at least one design")
        self.space = space
        self.designs = list(designs)
        self.budget_range = tuple(budget_range) if budget_range else BUDGET_RANGE
        self.rng = np.random.default_rng([seed, 11])
        self.totals: Dict[str, float] = {"calls": 0, "designs": 0}

    @classmethod
    def from_file(cls, space, path: str, seed: int = 0, budget_range=None) -> "ReplayPlanner":
        return cls(space, json.loads(Path(path).read_text()), seed, budget_range)

    def plan(self, contexts: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.totals["calls"] += 1
        self.totals["designs"] += len(contexts)
        out = []
        for _ in contexts:
            d = self.designs[int(self.rng.integers(len(self.designs)))]
            t = d.get("test") or {}
            out.append(
                {
                    "region": self.space.region(bounds=d.get("bounds"), picks=d.get("picks")),
                    "rule": str(d.get("rule", "adko")),
                    "budget": int(d.get("budget", self.budget_range[0])),
                    "hypothesis": "replayed design",
                    "rationale": str(d.get("origin", "replay")),
                    "test": Test(
                        metric=str(t.get("metric", "new_families")),
                        op=str(t.get("op", ">=")),
                        threshold=float(t.get("threshold", 1.0)),
                    ),
                    "p_success": 0.5,
                    "source": "replay",
                }
            )
        return out

    def record_outcome(self, campaign, verdict) -> None:
        return None

    def close(self) -> None:
        return None


def run(
    problem: Problem,
    surrogate_factory: Callable[[], Any],
    cfg: RunConfig,
    planner: Optional[Any] = None,
    out: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run one arm and return its record; write it to ``out`` when given."""
    t0 = time.time()
    ids = [f"Agent{i}" for i in range(cfg.agents)]
    topo = build_topology(cfg.topology, ids, **cfg.topology_kwargs)
    nbrs = {i: [ids.index(n) for n in topo.in_neighbors(ids[i])] for i in range(cfg.agents)}
    space = problem.region_space

    if planner is None and cfg.planner == "random":
        anchor = None
        if cfg.random_center == "pool":
            anchor = lambda rng: space.normalize(problem.pool(rng, 1).F)[0]  # noqa: E731
        elif cfg.random_center != "uniform":
            raise ValueError(f"random_center must be 'uniform' or 'pool', got {cfg.random_center!r}")
        planner = RandomPlanner(
            space, cfg.seed, cfg.agent_cfg.budget_range, anchor=anchor, widths=cfg.random_widths
        )
    elif planner is None and cfg.planner == "replay":
        if not cfg.replay_from:
            raise ValueError("the replay planner needs replay_from: a JSON list of designs")
        planner = ReplayPlanner.from_file(space, cfg.replay_from, cfg.seed, cfg.agent_cfg.budget_range)

    start, start_out, scale = _shared_start(problem, cfg)
    centers = space.centers(cfg.agents, cfg.seed) if cfg.agent_cfg.territory else None
    agents = [
        ADKOHAgent(
            ids[i],
            i,
            surrogate_factory(),
            problem,
            cfg.agent_cfg,
            neighbor_indices=nbrs[i],
            degree=max(len(nbrs[i]), 1),
            seed=cfg.seed,
            planner=None if cfg.planner == "none" else planner,
            centers=centers,
            descriptor_scale=scale,
        )
        for i in range(cfg.agents)
    ]
    if len(start_out):
        for a in agents:
            a.load(start, start_out)

    rounds: List[Dict[str, Any]] = []
    for rnd in range(cfg.rounds):
        _plan_round(agents, rnd, planner, cfg)
        picks: List[Pick] = []
        taken: set = set()
        for a in agents:
            p = a.propose(rnd, taken)
            taken.add(p.key)
            picks.append(p)
        outs = problem.evaluate(picks)
        if len(outs) != len(picks):
            raise RuntimeError(f"evaluator returned {len(outs)} outcomes for {len(picks)} picks")
        tokens = [a.complete(p, o, rnd) for a, p, o in zip(agents, picks, outs)]
        for i, a in enumerate(agents):
            a.observe([tokens[j] for j in nbrs[i]])
        rounds.append(_round_record(rnd, agents, outs, picks, planner))
        if verbose:
            r = rounds[-1]
            print(
                f"round {rnd:3d}  families={r['families']:3d}  hull={r['hull']:.3f}  "
                f"best={r['best']:.4f}  calls={r['llm_calls']}",
                flush=True,
            )

    record = _finish(problem, cfg, topo, agents, rounds, planner, time.time() - t0)
    if out:
        _write(Path(out), record, rounds, agents)
    return record


def _shared_start(problem: Problem, cfg: RunConfig):
    """The shared initial points, their outcomes, and the run's frozen descriptor scale.

    The scale is fixed here, once, from data every agent has: a descriptor quantized against a
    scale that drifted would mean different things on different tokens.
    """
    start = problem.start(cfg.seed, cfg.start_points)
    outs = []
    if len(start):
        outs = problem.evaluate(
            [Pick(agent=-1, key=int(k), X=start.X[i], F=start.F[i], meta=start.meta[i] if i < len(start.meta) else None)
             for i, k in enumerate(start.key)]
        )
    dim = problem.descriptor_dim()
    if outs and dim:
        Z = np.asarray([np.asarray(o.descriptor, dtype=float).ravel() for o in outs], dtype=float)
        scale = np.maximum(np.max(np.abs(Z), axis=0), 1e-9) if Z.size else np.ones(dim)
    else:
        scale = np.ones(dim) if dim else None
    return start, outs, scale


def _plan_round(agents: Sequence[ADKOHAgent], rnd: int, planner, cfg: RunConfig) -> None:
    """One batched planning call for every agent that needs a campaign this round."""
    need = [a for a in agents if a.needs_plan(rnd)]
    if need and planner is not None:
        designs = planner.plan([a.plan_context(rnd) for a in need])
        for a, d in zip(need, designs):
            a.set_campaign(d, rnd)


def _round_record(rnd, agents, outs, picks, planner) -> Dict[str, Any]:
    fams = {f for a in agents for f in a._fams if f}
    Z = _team_descriptors(agents)
    scores = [float(o.score) for o in outs]
    return {
        "round": rnd,
        "evaluated": sum(len(a._u) for a in agents),
        "families": len(fams),
        "hull": hmetrics.hull_volume(Z),
        "best": max(max(a._u) for a in agents) if any(a._u for a in agents) else 0.0,
        "round_best": max(scores) if scores else 0.0,
        "rules": sorted({p.rule for p in picks}),
        "llm_calls": int((getattr(planner, "totals", {}) or {}).get("calls", 0)),
        "claims": int(sum(a.claims.totals["claims_new"] for a in agents)),
        "verdicts": int(sum(a.claims.totals["verdicts_new"] for a in agents)),
    }


def _team_descriptors(agents) -> np.ndarray:
    rows = [z for a in agents for z in a._desc if z.size]
    if not rows:
        return np.zeros((0, 0))
    d = max(len(z) for z in rows)
    return np.asarray([z for z in rows if len(z) == d], dtype=float)


def _finish(problem, cfg, topo, agents, rounds, planner, secs) -> Dict[str, Any]:
    """Endpoints and run metadata, computed once from what the agents recorded."""
    recs = [r for a in agents for r in a.records]
    recs.sort(key=lambda r: (r["round"], r["agent"]))
    fams = [r["family"] for r in recs]
    scores = [r["score"] for r in recs]
    Z = _team_descriptors(agents)
    judged = [c for a in agents for c in a.campaigns if c.verdict]
    picked = [x for a in agents for x in a.own_picks]
    X = np.asarray(picked, dtype=float) if picked else np.zeros((0, 1))
    return {
        "label": cfg.label or f"{problem.problem_id}-{cfg.planner}",
        "problem": problem.problem_id,
        "config": _plain(cfg),
        "topology": topo.describe(),
        "seconds": round(secs, 2),
        "endpoints": {
            "evaluations": len(recs),
            "distinct_families": hmetrics.cumulative_distinct(fams)[-1] if fams else 0,
            "n_circles": hmetrics.n_circles(Z, cfg.circle_threshold),
            "hull_volume": hmetrics.hull_volume(Z),
            "auc_top10": hmetrics.auc_top_k(scores, 10),
            "best": max(scores) if scores else 0.0,
            "diverse_hits": hmetrics.diverse_hits(scores, fams, cfg.hit_threshold),
            "redundancy": hmetrics.redundancy(X, 1e-6),
            # Experiments are counted when run, so one re-run by another agent costs budget;
            # PMO counts distinct oracle calls. The two differ by the federation's repeats.
            "distinct_keys": len({r["key"] for r in recs}),
            "skipped_reported": int(sum(r.get("d_skipped", 0) or 0 for r in recs)),
            "load_balance": hmetrics.load_balance([r["agent"] for r in recs]),
        },
        "campaigns": {
            "n": len(judged),
            "passed": int(sum(1 for c in judged if c.verdict["success"])),
            "calibration": hmetrics.calibration(
                [c.p_success for c in judged], [bool(c.verdict["success"]) for c in judged]
            ),
            "mean_budget": float(np.mean([c.budget for c in judged])) if judged else None,
            "rules": {
                r: int(sum(1 for c in judged if c.rule == r)) for r in sorted({c.rule for c in judged})
            },
        },
        "planner": dict(getattr(planner, "totals", {}) or {}),
        "bits_per_token": float(np.mean([r["bits"] for r in recs])) if recs else 0.0,
        "rounds": len(rounds),
    }


def _plain(obj) -> Any:
    """Config as JSON, with the dataclasses inside it unpacked."""
    if is_dataclass(obj):
        return {k: _plain(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def _write(out: Path, record, rounds, agents) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(
        json.dumps({**record, "agents": [a.trace() for a in agents]}, indent=2, default=str)
    )
    with (out / "rounds.jsonl").open("w") as fh:
        for r in rounds:
            fh.write(json.dumps(r, default=str) + "\n")
    recs = [r for a in agents for r in a.records]
    recs.sort(key=lambda r: (r["round"], r["agent"]))
    cols = ["round", "agent", "key", "point", "score", "family", "rule", "claim_id", "bits"]
    with (out / "records.csv").open("w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in recs:
            fh.write(",".join(_csv(r.get(c)) for c in cols) + "\n")


def _csv(v) -> str:
    s = "" if v is None else str(v)
    return f'"{s}"' if any(ch in s for ch in ',"\n') else s
