"""All agents in one process. The control backend."""

from __future__ import annotations

from typing import Dict, List, Sequence

from appfl.decentralized.exchange.base import TokenExchange
from appfl.decentralized.protocol import TokenProtocol
from appfl.decentralized.topology import Topology


class InProcessExchange(TokenExchange):
    """All agents in one process. For tests, and for reproducing the reference exactly.

    The reference implementation is in-process, so this backend is the one that should
    reproduce its published numbers. The other two must then match *this*, which makes it
    the control in any "did the port survive distribution?" comparison.

    Publishing stages tokens rather than delivering them; :meth:`barrier` hands them over.
    That deferral is not an implementation detail -- it is what makes a round a clean step.
    Delivering immediately would let the second agent in a loop read the first agent's token
    from the round it is still in, which no other backend does and the reference does not do.
    Keeping the deferral *here* rather than in the driver is what lets one runner body be
    correct under all three transports.
    """

    owns_single_agent = False

    def __init__(self, topology: Topology, budget=None, meter=None, token_cls=None):
        super().__init__(topology, budget, meter, token_cls)
        self._inboxes: Dict[str, List[TokenProtocol]] = {
            a: [] for a in topology.agent_ids
        }
        self._pending: Dict[str, List[TokenProtocol]] = {
            a: [] for a in topology.agent_ids
        }

    def publish(self, agent_id: str, tokens: Sequence[TokenProtocol]) -> None:
        admitted = self._admit(tokens)
        for peer in self.topology.neighbors(agent_id):
            self._pending[peer].extend(admitted)
            self.meter.record_sent(admitted)

    def collect(self, agent_id: str) -> List[TokenProtocol]:
        received, self._inboxes[agent_id] = self._inboxes[agent_id], []
        return received

    def barrier(self, round_idx: int = 0) -> None:
        """Deliver everything staged this round. The in-process analogue of MPI's ``recv``."""
        for agent_id, tokens in self._pending.items():
            if tokens:
                self._inboxes[agent_id].extend(tokens)
                tokens.clear()
