"""The pluggable components ADKO's agent is built from, and does not itself supply.

Three of them, unrelated to each other except that the agent needs all three: an
uncertainty-aware :class:`Surrogate`, an optional :class:`LanguageModel`, and the
:class:`DesignSpace` being searched. "Component" is APPFL's own word for a plug-and-play part
behind a base class -- the same relationship an aggregator or a trainer has to the framework.

These are Bayesian-optimization concepts, not decentralization concepts, which is why they
live here rather than beside :class:`~appfl.decentralized.protocol.AgentProtocol`. A different
decentralized algorithm would need entirely different components and still use the same
transport, graph and round driver.

This is the file to hand to whoever owns the science: implement these, and the rest of the
package carries the result to the other agents.
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
        """Return ``(mu, sigma)`` per candidate. Batched: this is the per-round bottleneck."""

    @abstractmethod
    def update(self, embedding: Sequence[float], observation: float) -> None:
        """Algorithm 1 step 12: append to ``D_i`` and refit."""


class LanguageModel(ABC):
    """``M_i`` -- used at exactly two points in Algorithm 1, and optional at both.

    The paper ablates it: the NAS study runs with no LM at all, isolating token-based
    collaboration from semantic reasoning. Any port should keep that switch, because it is
    what separates "the tokens carry signal" from "the LM is doing the work".
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
        """Algorithm 1 step 3: propose ``n`` candidate *design points*.

        :param history: this agent's own ``(point, observation)`` pairs. Local and private --
            it goes to the agent's own model, not to peers.
        :param progress: ``n_obs``, ``best_y``, ``rounds_since_improve``,
            ``recent_improvement``.

        Both are needed for the model to *exploit* rather than only react to peers. The
        reference prompt carries a laboratory profile, the allowed options, a progress block,
        a coverage map, the lab's own observation memory, and peer evidence
        (``llm_suzuki.py::_propose_prompt``); without history and progress the model is
        working from peer tokens alone and cannot tell a promising region it has already
        exhausted from one it has never touched.

        Returns points in the space's own representation, not embeddings -- the agent calls
        ``space.embed`` on whatever comes back.

        This is where cross-slice transfer happens: the LM reads that an iodide/BPin motif
        worked in DMF and proposes its analogue in MeCN. In the paper's chemistry study this
        cuts the scored candidate set from ~3,696 to 10 per round at comparable hit rate.

        Returning an empty list must be safe. A flaky endpoint should degrade the run to the
        LM-free path, not end it.
        """

    @abstractmethod
    def encode_insight(
        self, embedding: Sequence[float], observation: float, threshold: float
    ) -> Optional[str]:
        """Algorithm 1 step 10: the natural-language ``z`` carried by an outgoing token.

        Returning ``None`` is legitimate and cheap -- ``z`` dominates token size, so an agent
        under a tight bit budget should emit numeric-only tokens.
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
        """Every point this agent may choose, or ``None`` if the space is not enumerable.

        The reference scores the *entire unobserved set* each round -- ~3,696 conditions in
        the Suzuki study, already restricted to this laboratory's solvent
        (``run_suzuki.py::_unobserved_candidates``). Reproducing its results requires the same
        pool, so a categorical space must return it here.

        Continuous spaces return ``None``; the agent then falls back to sampling, which is a
        deliberate approximation rather than the reference algorithm.
        """
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
