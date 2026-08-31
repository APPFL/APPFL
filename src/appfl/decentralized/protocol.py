"""What the transport layer requires, and nothing more.

Two contracts live here, and both are deliberately minimal -- they are the surface a
*generic* decentralized runtime needs, with no Bayesian optimization or ADKO in sight:

* :class:`TokenProtocol` -- what a payload must offer to be routed and metered.
* :class:`AgentProtocol` -- what the round driver needs to drive something.

Algorithm-specific interfaces (surrogate, language model, design space) live in
``appfl.decentralized.algorithm.adko.components``. Keeping them apart is what lets a second algorithm
reuse ``exchange.py``, ``runner.py``, ``topology.py`` and ``budget.py`` unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable


@runtime_checkable
class TokenProtocol(Protocol):
    """Everything the transport and metering layers touch on a payload.

    Measured, not guessed: ``exchange.py``, ``budget.py`` and ``runner.py`` between them use
    exactly ``serialize()``, ``size_bits()`` and ``provenance.agent_id``. They never look at
    what the token *means* -- the signal, the advantage, the embedding are all invisible to
    them, which is precisely why the same transport can carry a different algorithm's tokens.
    """

    provenance: Any  # must expose .agent_id for routing and .round for logging

    def serialize(self) -> bytes:
        """Bytes for the wire."""
        ...

    def size_bits(self) -> int:
        """Size against the communication budget."""
        ...


class AgentProtocol(ABC):
    """Minimal interface the APPFL runtime drives. ``ADKOAgent`` is the reference impl.

    One round is ``observe`` then ``act``. Splitting them lets the runtime interleave
    delivery and execution differently under MPI (bulk-synchronous) and gRPC (asynchronous)
    without the agent knowing which it is running under.
    """

    agent_id: str

    @abstractmethod
    def observe(self, tokens: Sequence[TokenProtocol]) -> None:
        """Ingest neighbor tokens. Must tolerate duplicates and out-of-order arrival.

        Typed against :class:`TokenProtocol` rather than a concrete token: the runtime hands
        over whatever the transport decoded, and an implementation narrows that to its own
        token type. ``ADKOAgent`` annotates these as ``KnowledgeToken``, which is a legitimate
        narrowing -- an agent knows what it is receiving even though the driver does not.
        """

    @abstractmethod
    def act(self, round_idx: int) -> List[TokenProtocol]:
        """Run one round; return tokens to broadcast (at most one per ADKO Constraint 3.2)."""

    def state_dict(self) -> Dict[str, Any]:
        """For checkpointing long HPC runs. Default: not checkpointable."""
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        return None

    def best_so_far(self) -> Optional[Tuple[Any, float]]:
        """``(design_point, observation)``. Read by the evaluation harness, never by peers."""
        return None
