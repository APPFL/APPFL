"""Campaigns: a region, a rule, a budget, and a test the planner can fail.

ADKO-H moves the language model up one level. In L3 it replaced a single pick, competing with
a calibrated score on the score's own ground, and the screen found it costs more than it adds.
A campaign is the unit where a model can plausibly beat a score instead: *where to look, how
to look there, for how long, and what would count as being right*. Every individual experiment
inside the campaign is still chosen by a numeric rule.

The test is what makes this more than prompting. The planner states a hypothesis, a budget and
a threshold up front, the campaign runs, and the verdict is computed from the data -- not from
the model's opinion of its own work. The verdict travels to neighbours, and the stated
``p_success`` makes the planner scoreable (Brier against a constant predictor), so "the LLM
helped" becomes a measurement rather than a vibe.

The tests are novelty-first, because the applications this is built for reward *distinct*
outcomes rather than one optimum: how many unseen family keys a region yields, how far its best
point lands from everything already known, how much it grows the descriptor hull. Objective
tests are kept for yield-style proxies.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

#: Rules a campaign may ask for; see ``rules.py``.
RULES = ("adko", "indep", "exploit", "explore", "novelty")
#: Default smallest and largest budget a campaign may claim, in experiments. A campaign is
#: the unit a language model is paid for, so this is what sets the token cost of a run: at a
#: 10,000-experiment budget, campaigns of 5 mean 2,000 calls and campaigns of 40 mean 250.
#: Arms set it from the run's own size (``AgentConfig.budget_range``).
BUDGET_RANGE = (2, 10)


@dataclass
class Evidence:
    """What a test is judged on: the campaign's own runs, against what was known at its start.

    The agent fills this in; a test never reaches into the agent. ``known_*`` and
    ``hull_before`` are snapshotted when the campaign starts, so a test cannot be satisfied by
    something a neighbour found while the campaign was running.
    """

    scores: np.ndarray = field(default_factory=lambda: np.zeros(0))
    descriptors: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    families: List[str] = field(default_factory=list)
    known_descriptors: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    known_families: Set[str] = field(default_factory=set)
    hull_before: float = 0.0
    hull_after: float = 0.0
    #: What counts as a hit, for the hit metrics; ``None`` means nothing does.
    hit_threshold: Optional[float] = None
    maximize: bool = True

    def hit_mask(self) -> np.ndarray:
        """Which of the campaign's outcomes are hits: at or past the threshold."""
        s = np.asarray(self.scores, dtype=float)
        if self.hit_threshold is None or s.size == 0:
            return np.zeros(s.size, dtype=bool)
        return s >= self.hit_threshold if self.maximize else s <= self.hit_threshold


def _new_families(ev: Evidence) -> float:
    return float(len({f for f in ev.families if f} - ev.known_families))


def _hits(ev: Evidence) -> float:
    return float(np.count_nonzero(ev.hit_mask()))


def _new_hit_families(ev: Evidence) -> float:
    """Families among the campaign's hits that nobody had reported when it started.

    The campaign's own increment to a diverse-hits endpoint, which ``new_families`` is not:
    on a large pool almost every experiment finds a new family, and almost none is a hit.
    """
    hit = ev.hit_mask()
    return float(len({f for f, h in zip(ev.families, hit) if h and f} - ev.known_families))


def _novel_distance(ev: Evidence) -> float:
    """How far the campaign's most isolated point sits from everything known at its start."""
    Z, K = np.atleast_2d(ev.descriptors), np.atleast_2d(ev.known_descriptors)
    if Z.size == 0:
        return 0.0
    if K.size == 0 or K.shape[1] != Z.shape[1]:
        return float("inf")
    d = np.sqrt(((Z[:, None, :] - K[None, :, :]) ** 2).sum(axis=2))
    return float(np.max(np.min(d, axis=1)))


def _hull_growth(ev: Evidence) -> float:
    return float(ev.hull_after - ev.hull_before)


def _max_objective(ev: Evidence) -> float:
    return float(np.max(ev.scores)) if np.size(ev.scores) else 0.0


def _mean_objective(ev: Evidence) -> float:
    return float(np.mean(ev.scores)) if np.size(ev.scores) else 0.0


#: Metric name -> how it is computed from :class:`Evidence`.
TESTS: Dict[str, Callable[[Evidence], float]] = {
    "new_families": _new_families,
    "novel_distance": _novel_distance,
    "hull_growth": _hull_growth,
    "max_objective": _max_objective,
    "mean_objective": _mean_objective,
    "hits": _hits,
    "new_hit_families": _new_hit_families,
}
OPS = (">=", "<=")


@dataclass(frozen=True)
class Test:
    """``metric op threshold`` -- the whole of what the planner commits to.

    Deliberately tiny. A richer language would let the planner write tests it cannot fail,
    which is the failure mode this design exists to avoid.
    """

    metric: str = "new_families"
    op: str = ">="
    threshold: float = 1.0

    def __post_init__(self) -> None:
        if self.metric not in TESTS:
            raise ValueError(f"metric must be one of {sorted(TESTS)}, got {self.metric!r}")
        if self.op not in OPS:
            raise ValueError(f"op must be one of {OPS}, got {self.op!r}")
        if not math.isfinite(self.threshold):
            raise ValueError("threshold must be finite")

    def value(self, ev: Evidence) -> float:
        return TESTS[self.metric](ev)

    def passed(self, ev: Evidence) -> bool:
        v = self.value(ev)
        return v >= self.threshold if self.op == ">=" else v <= self.threshold

    def label(self) -> str:
        return f"{self.metric} {self.op} {self.threshold:g}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Campaign:
    """One planned stretch of a run: what was committed to, and what came of it.

    Args:
        region: where to look.
        rule: how to pick inside it (one of :data:`RULES`).
        budget: how many experiments to spend before judging.
        hypothesis: one line, stated before the data exists.
        rationale: why this region and rule, in the planner's own words.
        note: one sentence for the neighbours, written when this campaign opened: what the
            previous one taught. Only the detailed channel sends it.
        test: what would count as being right.
        p_success: the planner's stated probability that the test passes.
        claim_id: this agent's id for the campaign, ``a{source}c{n}``; what a verdict cites.
        started: round the campaign began.
    """

    region: Any
    rule: str = "adko"
    budget: int = 5
    hypothesis: str = ""
    rationale: str = ""
    note: str = ""
    test: Test = field(default_factory=Test)
    p_success: float = 0.5
    claim_id: str = ""
    started: int = 0
    runs: List[Dict[str, Any]] = field(default_factory=list)
    baseline: Dict[str, Any] = field(default_factory=dict)
    verdict: Optional[Dict[str, Any]] = None
    source: str = "llm"
    budget_range: Tuple[int, int] = BUDGET_RANGE

    def __post_init__(self) -> None:
        self.rule = self.rule if self.rule in RULES else "adko"
        if self.source != "none":
            # A planner's budget is clamped; the no-planner arm's single campaign is
            # open-ended by design -- it is plain ADKO, which never stops to be judged.
            lo, hi = self.budget_range
            self.budget = int(min(max(int(self.budget), lo), hi))
        self.p_success = float(min(max(float(self.p_success), 0.0), 1.0))

    @property
    def spent(self) -> int:
        return len(self.runs)

    @property
    def done(self) -> bool:
        return self.spent >= self.budget

    @property
    def last(self) -> bool:
        """Whether the next experiment is the campaign's last -- when the verdict is written."""
        return self.spent + 1 >= self.budget

    def judge(self, ev: Evidence) -> Dict[str, Any]:
        """Close the campaign: compute the metric, compare it, keep the result.

        Idempotent, so a replayed run does not re-judge.
        """
        if self.verdict is None:
            v = self.test.value(ev)
            self.verdict = {
                "claim_id": self.claim_id,
                "metric": self.test.metric,
                "value": None if not math.isfinite(v) else round(float(v), 6),
                "op": self.test.op,
                "threshold": float(self.test.threshold),
                "success": bool(self.test.passed(ev)),
                "p_success": self.p_success,
                "spent": self.spent,
                "hypothesis": self.hypothesis,
            }
        return self.verdict

    def brier(self) -> Optional[float]:
        """``(p - outcome)^2`` once judged: the planner's calibration, one campaign at a time."""
        if self.verdict is None:
            return None
        return float((self.p_success - (1.0 if self.verdict["success"] else 0.0)) ** 2)

    def to_dict(self, space=None) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "region": self.region.to_dict() if hasattr(self.region, "to_dict") else {},
            "region_label": self.region.label(space) if space is not None else "",
            "share": round(self.region.share(space), 6) if space is not None else None,
            "rule": self.rule,
            "budget": self.budget,
            "spent": self.spent,
            "hypothesis": self.hypothesis,
            "rationale": self.rationale,
            "note": self.note,
            "test": self.test.to_dict(),
            "p_success": self.p_success,
            "started": self.started,
            "source": self.source,
            "verdict": self.verdict,
        }


def random_design(
    space,
    rng: np.random.Generator,
    half_width: Tuple[float, float] = (0.1, 0.4),
    budget_range: Tuple[int, int] = BUDGET_RANGE,
    center: Optional[Sequence[float]] = None,
    widths: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """A campaign design drawn at random -- the control arm's planner.

    Matched to the LLM arm in *lifecycle*, not in content: same call sites, same budget range,
    same tests, same claims and verdicts. Whatever the LLM arm wins over this is the planning,
    not the machinery around it.

    By default the box sits anywhere in the unit cube with one random half-width, which on a
    clumped pool leaves many boxes nearly empty. ``center`` (a unit-cube point, e.g. a pool
    member's normalized coordinates) and ``widths`` (one half-width per coordinate) draw boxes
    where candidates exist and shaped like a planner's, which is the control a planner's
    regions should be compared with.
    """
    center = rng.random(len(space)) if center is None else np.asarray(center, dtype=float)
    hw = float(rng.uniform(*half_width)) if widths is None else widths
    region = space.box_around(center, hw)
    metric = str(rng.choice(["new_families", "novel_distance", "hull_growth"]))
    threshold = {
        "new_families": float(rng.integers(1, 4)),
        "novel_distance": float(rng.uniform(0.5, 2.0)),
        "hull_growth": float(rng.uniform(0.0, 1.0)),
    }[metric]
    return {
        "region": region,
        "rule": str(rng.choice(RULES)),
        "budget": int(rng.integers(budget_range[0], budget_range[1] + 1)),
        "hypothesis": f"random design around {region.label(space)}",
        "rationale": "random control",
        "test": Test(metric=metric, op=">=", threshold=threshold),
        "p_success": 0.5,
        "source": "random",
    }
