"""Shared problem definition for the ADKO-on-APPFL demos.

Defined once so the in-process, MPI, and multi-site runners are running *the same
experiment*. That is the point of the exercise: if the three backends disagree on the result,
the distribution changed something it shouldn't have.

The science is a stand-in -- a closed-form "yield" function instead of a real experiment --
but the shape matches the reference implementation's Suzuki study: agents partition the design
space, each maximizes within its own slice, outcomes live on a 0-100 yield scale, and the
SUCCESS threshold is 50. Swap :class:`KernelSurrogate` for a real GP and ``true_yield`` for a
solver call and nothing else here changes.
"""

from __future__ import annotations

import math
import random
from typing import Any, List, Optional, Sequence, Tuple

from appfl.decentralized import Meter, Topology, build_topology, metered_evaluator
from appfl.decentralized.algorithm.adko import (
    ADKOAgent,
    ConfidencePruner,
    DesignSpace,
    FIFOPruner,
    FidelityAwarePruner,
    LLMConfig,
    RandomPruner,
    ReasoningWeights,
    Surrogate,
    build_baseline,
    build_language_model,
)

# Four agents partitioning the unit interval, mirroring the reference's four laboratories
# each restricted to its own solvent.
WINDOWS = [(0.00, 0.30), (0.25, 0.50), (0.45, 0.75), (0.70, 1.00)]
AGENT_IDS = [f"agent-{i}" for i in range(len(WINDOWS))]

# ADKO's tau and the advantage-score normalizer, on the 0-100 yield scale.
THRESHOLD = 50.0
SCALE = 50.0

# Two published configurations, both selectable. The v2 paper's recommended starting point
# is the many-task one (running median baseline, symmetric weights); the Suzuki study uses a
# fixed domain threshold and heavily asymmetric weights. Neither is "the" right answer -- the
# choice depends on whether "success" has a domain meaning, which is why it is a flag.
PRESETS = {
    "many_task": dict(
        weights=lambda: ReasoningWeights.many_task(dim=1),
        baseline=lambda: build_baseline("running_median"),
        pruner="confidence",
    ),
    "suzuki": dict(
        weights=lambda: ReasoningWeights(
            beta=2.0, lam=4.0, gamma=32.0, sigma_s=0.05,
            metric="euclidean", kernel="two_sigma_sq",
            weight_by_fidelity=True, peer_normalization="mixing_weight",
        ),
        baseline=lambda: build_baseline("fixed", threshold=THRESHOLD, scale=SCALE),
        pruner="fidelity",
    ),
}
DEFAULT_TOKEN_BUDGET = 40
DEFAULT_ALPHA_TAU = 0.01

PRUNERS = {
    "fidelity": lambda: FidelityAwarePruner(alpha_tau=DEFAULT_ALPHA_TAU),
    "confidence": lambda: ConfidencePruner(),
    "fifo": lambda: FIFOPruner(),
    "random": lambda: RandomPruner(),
}


def true_yield(x: float) -> float:
    """Stand-in for an expensive evaluation: a percentage yield in [0, 100].

    Peak ~100 at x=0.35, decoy ~40 at x=0.75. Only agent-0 and agent-1 own windows near the
    true peak, so agents 2 and 3 can only find it by acting on what a neighbor tells them.
    """
    return 100.0 * math.exp(-((x - 0.35) ** 2) / 0.01) + 40.0 * math.exp(
        -((x - 0.75) ** 2) / 0.02
    )


class Interval1D(DesignSpace):
    """A slice ``[lo, hi]`` of the unit interval. ``phi`` quantizes -- a crude privacy map.

    A real deployment uses DP noise (``appfl.privacy``) or the reference's randomized
    response. Quantization is enough to show the shape: a neighbor learns the region, not the
    recipe.
    """

    space_id = "toy-1d"

    def __init__(self, lo: float, hi: float, n_bins: int = 50, seed: int = 0):
        self.lo, self.hi = lo, hi
        self.n_bins = n_bins
        self.rng = random.Random(seed)

    def embed(self, point: Any) -> List[float]:
        return [round(point * self.n_bins) / self.n_bins]

    def sample(self, n: int, seed: Optional[int] = None) -> List[Any]:
        rng = random.Random(seed) if seed is not None else self.rng
        return [rng.uniform(self.lo, self.hi) for _ in range(n)]

    def enumerate(self) -> Optional[List[Any]]:
        """The quantization grid restricted to this agent's window.

        Enumerable on purpose: it puts this toy on the reference's code path -- score the
        whole unobserved set each round -- rather than the sampling fallback. The real
        Suzuki space is enumerable too (3,696 conditions), so this is the shape that matters.
        """
        step = 1.0 / self.n_bins
        lo = int(math.ceil(self.lo * self.n_bins))
        hi = int(math.floor(self.hi * self.n_bins))
        return [i * step for i in range(lo, hi + 1)]

    def local_perturbations(self, around: Any, n: int) -> List[Any]:
        return [
            min(self.hi, max(self.lo, around + self.rng.gauss(0, 0.05))) for _ in range(n)
        ]

    # -- hooks used only when an LLM is attached --------------------------------------

    def describe(self) -> str:
        """This agent's slice, in words. Deliberately does not mention the other slices."""
        return (
            f"A single continuous parameter x, which this laboratory may set anywhere in "
            f"[{self.lo:.3f}, {self.hi:.3f}] and nowhere else. Higher measured yield is "
            f"better; yields run 0-100."
        )

    def parse(self, payload: Any) -> Optional[Any]:
        """Accept a number or {"x": number} inside this slice; reject anything else.

        Rejecting rather than clamping is deliberate: a clamped candidate looks like a real
        proposal in the logs and quietly biases the arm toward the slice boundary.
        """
        if isinstance(payload, dict):
            payload = payload.get("x", payload.get("value"))
        try:
            x = float(payload)
        except (TypeError, ValueError):
            return None
        return x if self.lo <= x <= self.hi else None


class KernelSurrogate(Surrogate):
    """Nadaraya-Watson mean with distance-based uncertainty. Stands in for a GP.

    Dependency-free on purpose, so the demos run anywhere APPFL runs. It reproduces the two
    properties the reasoning score depends on: a mean that tracks observations, and a sigma
    that grows away from them so ``beta * sigma`` drives exploration.
    """

    def __init__(self, bandwidth: float = 0.05, prior_sigma: float = 1.0):
        self.bandwidth = bandwidth
        self.prior_sigma = prior_sigma
        self.xs: List[float] = []
        self.ys: List[float] = []

    def posterior(self, candidates: Sequence[Sequence[float]]) -> List[Tuple[float, float]]:
        """Return **standardized** ``(mu, sigma)``.

        Standardizing here is not cosmetic. The peer terms G and Lambda are normalized into
        [0, 1] by construction, so a posterior left on the raw 0-100 yield scale would swamp
        them no matter how lam and gamma are set. The reference standardizes for the same
        reason (``mu_std``, ``sigma_std`` in ``run_suzuki.py``), and the surrogate is the
        right place for it -- it is the only component that knows the objective's scale.
        """
        out = []
        spread = self._spread()
        center = sum(self.ys) / len(self.ys) if self.ys else 0.0
        for candidate in candidates:
            x = candidate[0]
            if not self.xs:
                out.append((0.0, self.prior_sigma))
                continue
            weights = [
                math.exp(-((x - xi) ** 2) / (2 * self.bandwidth**2)) for xi in self.xs
            ]
            total = sum(weights)
            mean = (
                sum(w * y for w, y in zip(weights, self.ys)) / total
                if total > 1e-12
                else 0.0
            )
            nearest = min(abs(x - xi) for xi in self.xs)
            sigma = self.prior_sigma * (
                1.0 - math.exp(-((nearest / self.bandwidth) ** 2))
            )
            out.append(((mean - center) / spread, sigma))
        return out

    def _spread(self) -> float:
        if len(self.ys) < 2:
            return 1.0
        center = sum(self.ys) / len(self.ys)
        var = sum((y - center) ** 2 for y in self.ys) / len(self.ys)
        return max(math.sqrt(var), 1e-8)

    def update(self, embedding: Sequence[float], observation: float) -> None:
        self.xs.append(embedding[0])
        self.ys.append(observation)


def make_topology(name: str = "fully_connected", **kwargs) -> Topology:
    """The federation graph. ``fully_connected`` is the ADKO reference default."""
    return build_topology(name, AGENT_IDS, **kwargs)


def make_agent(
    agent_id: str,
    topology: Topology,
    meter: Meter,
    *,
    preset: str = "many_task",
    lam: Optional[float] = None,
    gamma: Optional[float] = None,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    pruner_name: Optional[str] = None,
    seed: int = 0,
    warmup_rounds: int = 5,
    llm_config: Optional[LLMConfig] = None,
) -> ADKOAgent:
    """Build one agent. Called once per process under MPI and per site over gRPC.

    Every runner goes through here, so an agent is configured identically no matter which
    transport is carrying its tokens -- which is what makes cross-backend comparison mean
    anything.
    """
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; available: {sorted(PRESETS)}")
    index = AGENT_IDS.index(agent_id)
    lo, hi = WINDOWS[index]
    config = PRESETS[preset]

    weights = config["weights"]()
    # lam/gamma override the preset when given, so the no-communication arm (0, 0) and the
    # attraction-only / repulsion-only ablations work against either configuration.
    if lam is not None:
        weights.lam = lam
    if gamma is not None:
        weights.gamma = gamma
    pruner = PRUNERS[pruner_name or config["pruner"]]()
    return ADKOAgent(
        agent_id=agent_id,
        surrogate=KernelSurrogate(),
        space=Interval1D(lo, hi, seed=seed + index),
        evaluator=metered_evaluator(
            true_yield, meter, cost_seconds=1.0
        ),  # priced as if it were a real experiment
        # uniform_weight, not Metropolis-Hastings: 1/(|N_i|+1) is what the reference uses.
        mixing_weight=lambda other, me=agent_id: topology.uniform_weight(me, other),
        baseline=config["baseline"](),
        objective="maximize",
        # None unless an LLM is configured -- that is the LM-free ablation arm, which is
        # how the reference's NAS study runs.
        language_model=build_language_model(llm_config, agent_id) if llm_config else None,
        pruner=pruner,
        weights=weights,
        token_budget=token_budget,
        emit_insight=bool(llm_config and llm_config.emit_insight),
        warmup_rounds=warmup_rounds,
        seed=seed + index,
    )


def report(label: str, topology: Topology, agents: Sequence[ADKOAgent], meter: Meter,
           n_rounds: int) -> None:
    """Uniform result block, so output from the three backends can be diffed directly."""
    print(f"\n=== {label} ===")
    print(f"topology            : {topology.describe()}")
    example = agents[0] if agents else None
    if example is not None:
        w = example.weights
        print(
            f"config              : beta={w.beta} lam={w.lam} gamma={w.gamma} "
            f"sigma_s={w.sigma_s:.3f} kernel={w.kernel} "
            f"fidelity_weighted={w.weight_by_fidelity} norm={w.peer_normalization}"
        )
        print(f"baseline            : {type(example.baseline).__name__}")
    for agent in agents:
        found = agent.best_so_far()
        lo, hi = WINDOWS[AGENT_IDS.index(agent.agent_id)]
        if found is not None:
            print(
                f"  {agent.agent_id} owns [{lo:.2f},{hi:.2f}]  "
                f"best x={found[0]:.3f} yield={found[1]:.1f}  "
                f"eta_bar={agent.mean_token_fidelity():.3f}"
            )
    if meter.best_by_round:
        print(f"federation best     : {max(meter.best_by_round):.1f}  (true optimum 100.0)")
    print(f"tokens emitted      : {meter.tokens_emitted}")
    print(f"bits sent           : {meter.bits_sent}")
    print(f"bits per round      : {meter.bits_per_round(n_rounds):.0f}")
    print(f"evaluations         : {meter.evaluations}")
    trace = getattr(meter, "mean_fidelity_by_round", None)
    if trace:
        print(f"eta_bar (last round): {trace[-1]:.3f}")
    llm_stats = [
        a.language_model.stats()
        for a in agents
        if getattr(a, "language_model", None) is not None
    ]
    if llm_stats:
        total = {
            k: sum(s[k] for s in llm_stats) for k in ("calls", "cache_hits", "failures")
        }
        print(
            f"llm calls           : {total['calls']} "
            f"({total['cache_hits']} cached, {total['failures']} failed)"
        )
