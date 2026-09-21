"""Interfaces ADKO needs from a scientific application.

ADKO is domain-agnostic: the application supplies a surrogate model, a search
space, and optionally a language model. In Rillo et al. (ADKO), the surrogate is
usually a Gaussian process fitted only to one agent's local observations. ADKO
only needs it to predict an expected value ``mu(theta)`` and uncertainty
``sigma(theta)`` for embedded candidate points ``phi(theta)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

from appfl.decentralized.algorithm.adko.knowledge_token import KnowledgeToken


class Surrogate(ABC):
    """The agent's private uncertainty-aware model. ADKO uses a Matern-5/2 GP.

    Never leaves the agent, and neither do its hyperparameters -- Constraint 3.1 forbids
    transmitting "any GP parameter inferred from D_i, or any sufficient statistic of D_i".
    The interface deliberately offers no serialization method for that reason.
    """

    @abstractmethod
    def posterior(
        self, candidates: Sequence[Sequence[float]]
    ) -> List[Tuple[float, float]]:
        """Return ``(mu, sigma)`` for each candidate embedding."""

    @abstractmethod
    def update(self, embedding: Sequence[float], observation: float) -> None:
        """Update with one local pair ``(phi(theta), y)``."""


class LanguageModel(ABC):
    """Optional model used for candidate proposals and token insights.

    It can suggest new points from token memory and write the optional text
    field ``z`` in a token.
    """

    @abstractmethod
    def propose(
        self,
        token_memory: Sequence[KnowledgeToken],
        space: "DesignSpace",
        n: int,
        history: Optional[Sequence[Tuple[Any, float]]] = None,
        progress: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """Return candidate design points.

        Args:
            token_memory: Peer tokens the model can use.
            space: Search space for valid proposals.
            n: Maximum number of candidates.
            history: Optional local ``(point, y)`` history.
            progress: Optional progress summary.
        """

    @abstractmethod
    def encode_insight(
        self, embedding: Sequence[float], observation: float, threshold: float
    ) -> Optional[str]:
        """Return optional insight ``z`` for an outgoing token.

        Args:
            embedding: Evaluated point as ``phi(theta)``.
            observation: Local result ``y``.
            threshold: Baseline ``b`` for success or failure.
        """


class DesignSpace(ABC):
    """The shared compact design space ``Theta``, plus the embedding ``phi``.

    ``phi`` is the privacy boundary for locations: DP noise, quantization, or a learned
    projection. It must be applied before anything leaves the agent, and the *same* ``phi``
    must be used federation-wide or the similarity kernel compares incomparable vectors.
    """

    @abstractmethod
    def embed(self, point: Any) -> List[float]:
        """``phi(theta)``. Non-invertible."""

    @abstractmethod
    def sample(self, n: int, seed: Optional[int] = None) -> List[Any]:
        """Draw candidate design points, for the exploration perturbations in step 4."""

    def enumerate(self) -> Optional[List[Any]]:
        """Return all feasible points, or ``None`` if the space is not enumerable."""
        return None

    def local_perturbations(self, around: Any, n: int) -> List[Any]:
        """Algorithm 1 step 4: exploitation perturbations near the current best.

        Default falls back to uniform sampling; override for spaces where "near" is
        meaningful (continuous compositions) as opposed to categorical (ligand choice).
        """
        return self.sample(n)

    # -- hooks used only when a LanguageModel is attached ------------------------------

    def describe(self) -> str:
        """What this agent is allowed to choose, in words the LM can act on.

        Describe *this agent's slice*, not the global space -- an agent restricted to one
        solvent should not be offered the others. Only called when an LM is configured.
        """
        raise NotImplementedError(
            f"{type(self).__name__} needs describe() before it can be used with a "
            f"LanguageModel; it tells the model what this agent may choose."
        )

    def parse(self, payload: Any) -> Optional[Any]:
        """Turn one LM-proposed JSON value into a design point, or ``None`` to reject it.

        This is a trust boundary: the model will occasionally return values outside the
        agent's slice, wrong types, or prose. Reject rather than clamp -- a silently clamped
        candidate looks like a real proposal in the logs and quietly biases the arm.
        """
        raise NotImplementedError(
            f"{type(self).__name__} needs parse() before it can be used with a "
            f"LanguageModel; it validates what the model proposes."
        )
