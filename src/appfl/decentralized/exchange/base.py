"""The transport contract every backend implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

from appfl.decentralized.budget import CommBudget
from appfl.decentralized.exchange.codec import default_token_cls
from appfl.decentralized.metrics import Meter
from appfl.decentralized.protocol import TokenProtocol
from appfl.decentralized.topology import Topology


class TokenExchange(ABC):
    """Move knowledge tokens between agents according to a topology.

    Implementations enforce ADKO Constraint 3.2 -- at most one token of at most B bits per
    neighbor per round -- through :attr:`budget`, and record what actually crossed the wire
    in :attr:`meter`, so an MPI run on a cluster and a live multi-site run produce the same
    accounting.
    """

    def __init__(
        self,
        topology: Topology,
        budget: Optional[CommBudget] = None,
        meter: Optional[Meter] = None,
        token_cls: Optional[type] = None,
    ):
        """
        :param token_cls: the concrete token class to rebuild received bytes into. Defaults
            to ADKO's :class:`KnowledgeToken` for convenience; pass your own to carry a
            different algorithm's payload over the same transport. This is the only place
            the transport layer needs to know a concrete type at all.
        """
        self.topology = topology
        self.budget = budget or CommBudget()
        self.meter = meter or Meter()
        self.token_cls = token_cls if token_cls is not None else default_token_cls()

    @abstractmethod
    def publish(self, agent_id: str, tokens: Sequence[TokenProtocol]) -> None:
        """Send ``tokens`` to every neighbor of ``agent_id``, trimmed to the bit budget."""

    @abstractmethod
    def collect(self, agent_id: str) -> List[TokenProtocol]:
        """Return tokens addressed to ``agent_id`` since the last call, and clear them."""

    #: Whether one instance drives exactly one agent (MPI rank, remote site) or all of them
    #: (single process). ``run_local_agent`` and ``run_federation`` check this, so pairing a
    #: driver with the wrong transport fails loudly instead of producing quiet nonsense.
    owns_single_agent: bool = True

    def barrier(self, round_idx: int = 0) -> None:
        """Deliver this round's traffic and synchronize the end of ``round_idx``.

        Every backend defers delivery to here: MPI posts its receives, the relay waits for
        peers, the in-process exchange flushes its staging buffer. That uniformity is what
        lets a single runner body be correct under all three -- if delivery happened during
        ``publish`` in one backend and at the barrier in another, the same loop would produce
        different round semantics depending on the transport underneath it.

        The round index is supplied by the caller because a backend that has to infer it
        cannot tell a fresh barrier from a repeat poll.
        """

    def close(self) -> None:
        """Release transport resources."""

    def _admit(self, tokens: Sequence[TokenProtocol]) -> List[TokenProtocol]:
        """Apply Constraint 3.2 and record the drop, shared by every backend."""
        admitted = self.budget.admits(tokens)
        self.meter.tokens_dropped_budget += len(tokens) - len(admitted)
        return admitted
