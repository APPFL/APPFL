"""What ADKO-H needs from an application, and what it hands back.

ADKO v2's ``components.DesignSpace`` assumes an enumerable grid every agent can score whole,
because Suzuki has 3,696 conditions. Large problems are not like that: a molecule pool has
250k members, a simulator's inputs are continuous, and in both cases the interesting experiments
are *generated* each round rather than enumerated once. Three things change, and they are the
reason this is a sibling package rather than a subclass:

* **a pool per agent-round**, not a global grid. The problem draws it, so it can screen for
  feasibility (a cheap physics or chemistry screen) or sample a region;
* **batch evaluation**, because a round's picks go to the GPUs together. The round driver
  collects every agent's pick and calls :meth:`Problem.evaluate` once;
* **outcomes richer than a scalar**. A diversity endpoint needs a descriptor (where the result
  sits in outcome space) and a family key (which qualitative class it belongs to), and the
  campaign tests are written against exactly those.

A point is identified by an integer ``key``, unique within a run. The problem decides what it
means -- a pool index for molecules, a counter for freshly drawn continuous points -- and
everything above this layer only compares keys for equality, so nothing else has to care.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from appfl.decentralized.algorithm.adko_h.region import RegionSpace


@dataclass
class Candidates:
    """One agent's candidate pool for one round.

    Args:
        key: ``(n,)`` ids, unique within the run.
        X: ``(n, d)`` embedding -- what the surrogate is fit on and the peer kernel compares.
        F: ``(n, k)`` region coordinates, in the region space's column order.
        meta: whatever the problem needs to evaluate a row, row-aligned (SMILES, a raw vector).
    """

    key: np.ndarray
    X: np.ndarray
    F: np.ndarray
    meta: Sequence[Any] = ()

    def __len__(self) -> int:
        return len(self.key)

    def row(self, i: int) -> Tuple[int, np.ndarray, np.ndarray, Any]:
        return (
            int(self.key[i]),
            np.asarray(self.X[i], dtype=float),
            np.asarray(self.F[i], dtype=float),
            self.meta[i] if i < len(self.meta) else None,
        )


@dataclass
class Pick:
    """One agent's chosen experiment, on its way to the evaluator."""

    agent: int
    key: int
    X: np.ndarray
    F: np.ndarray
    meta: Any = None
    rule: str = "adko"
    claim_id: str = ""


@dataclass
class Outcome:
    """One finished experiment.

    Args:
        score: the scalar the surrogate and the token's ``q`` are built from.
        descriptor: where the result sits in outcome space -- the novelty and hull tests live
            here. Must be the same dimension and basis for every agent in a run.
        family: a qualitative class, or ``""`` when the problem has none. Distinct families are
            the count endpoint, and ``new_families`` tests against it.
        extras: anything the run record should keep (runtime, flags, failure reasons).
    """

    score: float
    descriptor: np.ndarray = field(default_factory=lambda: np.zeros(0))
    family: str = ""
    extras: Dict[str, Any] = field(default_factory=dict)


class Problem(ABC):
    """The application side of ADKO-H. One instance is shared by every agent in a run.

    Implementations live in ``problems/``. Nothing here knows about agents, tokens or language
    models, which is what keeps the proxy benchmark and the real simulation interchangeable.
    """

    #: Logged with every run; distinguishes runs on different problems.
    problem_id: str = ""
    #: The coordinates the planner reasons about.
    region_space: RegionSpace
    #: Whether larger scores are better. ``q`` and every rule follow this.
    maximize: bool = True
    #: Peer kernel for this embedding: ``"euclidean"`` or ``"hamming"``.
    kernel: str = "euclidean"

    @abstractmethod
    def pool(
        self,
        rng: np.random.Generator,
        n: int,
        exclude: Optional[Sequence[int]] = None,
        region=None,
    ) -> Candidates:
        """``n`` candidates for one agent-round, excluding keys already run.

        ``region`` is a hint, not a constraint: a problem that can sample inside a region
        cheaply (continuous spaces) should, and one that cannot (a fixed pool) ignores it and
        lets the rule mask instead.
        """

    @abstractmethod
    def evaluate(self, picks: Sequence[Pick]) -> List[Outcome]:
        """Run every pick of one round, in order. One outcome per pick.

        A failure is an outcome too -- the worst admissible score, with the reason in
        ``extras`` -- because a budget is spent whether or not the simulation converged, and
        the baselines count it that way.
        """

    def start(self, seed: int, n: int) -> Candidates:
        """The initial points every arm of a comparison shares, or an empty set.

        Shared initial data is what makes arms paired seed for seed; the reference comparisons
        load these rather than spending budget on them.
        """
        return Candidates(np.zeros(0, dtype=np.int64), np.zeros((0, 1)), np.zeros((0, len(self.region_space))))

    def describe(self) -> Dict[str, Any]:
        """Text for the planner's prompts: the task, the outcome, any physics or chemistry.

        Keys: ``task`` (what is being optimized and why), ``outcome`` (what a descriptor and a
        family key mean), ``hints`` (a few domain facts), ``dims`` (defaults to the region
        space's own description).
        """
        return {
            "task": "",
            "outcome": "",
            "hints": "",
            "dims": self.region_space.describe(),
        }

    def label(self, key: int, meta: Any = None) -> str:
        """A short human-readable name for a point, for logs and prompts."""
        return str(key)

    def privatize(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """The embedding as it goes onto a token. Identity by default, as in v2."""
        return np.asarray(X, dtype=float)

    def input_scale(self) -> Optional[np.ndarray]:
        """Per-coordinate spread of the embedding, the unit ``AgentConfig.token_noise`` is in.

        ``None`` means 1 for every coordinate. A problem whose coordinates have very different
        spreads should return them, so one noise level means the same thing on each.
        """
        return None

    def descriptor_dim(self) -> int:
        """Dimension of :attr:`Outcome.descriptor`; 0 when the problem has none."""
        return 0

    def floor(self) -> float:
        """The score a failed experiment gets."""
        return 0.0
