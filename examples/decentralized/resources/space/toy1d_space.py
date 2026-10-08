"""Design space for the toy 1-D yield experiment -- the *local data* of one agent.

This is the decentralized analogue of ``examples/resources/dataset``: it is loaded by path
from an agent's config, and what it returns is the one thing an agent never shares -- the
slice of the design space it is allowed to probe, and the quantization it applies before a
peer is told anything about a point.

The shape mirrors the reference ADKO Suzuki study, one dimension instead of five: agents
partition the search space, each maximizes within its own slice, outcomes live on a 0-100
yield scale, and the success threshold is 50. Swap this file for
``suzuki_space.py`` in the config and nothing else in the example changes.
"""

from __future__ import annotations

import math
import random
from typing import Any, List, Optional, Sequence, Tuple

from appfl.decentralized.algorithm.adko import DesignSpace

#: The four windows of the published toy configuration. Only ``agent-1``'s window contains
#: the true peak, which is the shape that makes a neighbour's token worth anything.
DEFAULT_WINDOWS: Tuple[Tuple[float, float], ...] = (
    (0.00, 0.30),
    (0.25, 0.50),
    (0.45, 0.75),
    (0.70, 1.00),
)


def even_windows(num_agents: int, overlap: float = 0.2) -> List[Tuple[float, float]]:
    """``num_agents`` equal slices of [0, 1], each widened by ``overlap`` of its own width.

    Used when the launcher is given an agent count other than four -- an MPI scaling run, for
    instance. Neighbouring windows must overlap or the federation is partitioned in the
    objective even though the graph is connected.
    """
    width = 1.0 / num_agents
    pad = overlap * width
    return [
        (max(0.0, i * width - pad), min(1.0, (i + 1) * width + pad))
        for i in range(num_agents)
    ]


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


def get_design_space(
    agent_index: int = 0,
    num_agents: int = len(DEFAULT_WINDOWS),
    windows: Optional[Sequence[Sequence[float]]] = None,
    n_bins: int = 50,
    seed: int = 0,
) -> Interval1D:
    """Entry point named by ``space_configs.space_name`` in an agent config.

    ``agent_index``, ``num_agents`` and ``seed`` are filled in by the launcher, exactly as
    APPFL's FL launchers fill ``dataset_kwargs.client_id``.
    """
    if windows is None:
        windows = (
            DEFAULT_WINDOWS
            if num_agents == len(DEFAULT_WINDOWS)
            else even_windows(num_agents)
        )
    if not 0 <= agent_index < len(windows):
        raise ValueError(
            f"agent_index {agent_index} outside the {len(windows)} configured windows"
        )
    lo, hi = windows[agent_index]
    return Interval1D(float(lo), float(hi), n_bins=n_bins, seed=seed + agent_index)
