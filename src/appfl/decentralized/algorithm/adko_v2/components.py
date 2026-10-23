"""Interfaces ADKO v2 needs from a scientific application.

As in v1, ADKO v2 is domain-agnostic: the application supplies a surrogate, a design space and
an evaluator, and optionally a planner (the L3 language-model module). What v2 needs from each
is slightly different from v1, because the method changed:

* the surrogate reports **standardized** ``(mu, sigma)`` arrays -- the score adds them to a
  social term calibrated against ``sigma`` -- and can map a standardized mean back to the
  outcome scale, so trust can encode a prediction exactly as it encodes a result;
* the space must be **enumerable**: every round the agent scores its whole unobserved grid and
  calibrates the social term on the full grid, as the reference does;
* the space owns the privacy channel that turns ``phi(theta)`` into the shared input ``x``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class Surrogate(ABC):
    """The agent's private model, e.g. a GP. Never leaves the agent (Constraint 3.1)."""

    @abstractmethod
    def update(self, embedding: Sequence[float], observation: float) -> None:
        """Add one local pair ``(phi(theta), y)`` and refit."""

    @abstractmethod
    def predict(self, X: Sequence[Sequence[float]]) -> Tuple[np.ndarray, np.ndarray]:
        """Standardized posterior mean and SD at the rows of ``X``; ``(0, 1)`` before a fit."""

    @abstractmethod
    def raw_mean(self, mu: Sequence[float]) -> np.ndarray:
        """A standardized mean back on the outcome scale."""

    @property
    @abstractmethod
    def fitted(self) -> bool:
        """Whether a model exists yet. Trust learns only once it does."""


class DesignSpace(ABC):
    """One agent's slice ``Theta_i`` of the design space, and the embedding ``phi``.

    ``phi`` and the privacy channel must be the same federation-wide, or the similarity kernel
    compares incomparable vectors.
    """

    #: Logged with every token; distinguishes runs on different spaces.
    space_id: str = ""

    @abstractmethod
    def enumerate(self) -> List[Any]:
        """Every point this agent may run, in a fixed order -- its candidate grid."""

    @abstractmethod
    def embed(self, point: Any) -> List[float]:
        """``phi(theta)``: the surrogate's input, and the kernel's."""

    def grid(self) -> np.ndarray:
        """The enumerated points, embedded, as one ``(n, d)`` array (row ``r`` embeds point ``r``).

        The agent scores and calibrates on this array as it is, so its memory order decides
        the order BLAS sums the peer field in. A space that must agree bit for bit with another
        implementation returns the array that implementation scores on.
        """
        return np.asarray([self.embed(p) for p in self.enumerate()], dtype=float)

    def privatize(
        self, embedding: Sequence[float], rng: np.random.Generator
    ) -> List[Any]:
        """The input as sent, ``x = channel(phi(theta))``. Identity by default.

        Applied only to outgoing tokens; the agent's own surrogate keeps the true point.
        """
        return list(embedding)

    def warmup_points(self, seed: int, rounds: int) -> Optional[List[Any]]:
        """This agent's points for the first ``rounds`` rounds, or ``None`` to draw at random.

        A shared warmup lets runs that differ only after warmup be paired seed for seed, which
        is how the reference compares arms.
        """
        return None

    # -- hooks used only when a planner (L3) is attached -------------------------------

    def allowed(self) -> List[List[int]]:
        """Per-dimension option indices this agent may choose. Categorical spaces only."""
        raise NotImplementedError(
            f"{type(self).__name__} needs allowed() before it can be used with a planner; "
            f"it tells the planner what this agent may run."
        )

    def llm_context(self) -> Dict[str, Any]:
        """Anything else the planner's prompts need about this slice, e.g. the setting."""
        return {}


class Planner(ABC):
    """Optional module that sometimes chooses the next experiment instead of the score.

    What the agent calls. The L3 policies in ``llm/`` are the reference ones and implement
    this surface duck-typed, exactly as the reference defines it: with probability ``alpha``
    per agent-round after warmup a language model picks the experiment; otherwise the score
    does. The score always runs, so a rejected pick costs nothing but the call.
    """

    #: Call accounting for the report (calls, cache hits, errors, cost, ...).
    totals: Dict[str, float]

    @abstractmethod
    def wants(self, round_idx: int, agent: int) -> bool:
        """Whether this planner decides agent ``agent``'s experiment this round."""

    @abstractmethod
    def decide(self, contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """One ``{"x", "source", "insight", "call_id"}`` per context; ``x`` is ``None`` when
        there is no pick. A batch, as in the reference; an APPFL agent passes its own only."""

    def record_outcome(self, call_id: str, y: float, q: Optional[float]) -> None:
        """The result of an experiment this planner chose, for its own later prompts."""

    def close(self) -> None:
        """Flush logs and release clients."""
