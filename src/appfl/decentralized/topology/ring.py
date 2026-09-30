"""A cycle: each node exchanges with k neighbors either side."""

from __future__ import annotations

from typing import List, Sequence

from appfl.decentralized.topology.base import Topology


class Ring(Topology):
    """Each node talks to ``k`` neighbors either side. Cheap, low connectivity, slow mixing.

    The stress case: information needs O(n/k) rounds to cross the federation, so a ring is
    where DFL visibly stops behaving like FedAvg. Communication cost per node is constant in
    ``n``, which is exactly why it is worth the slower convergence at scale.
    """

    def __init__(self, node_ids: Sequence[str], k: int = 1):
        super().__init__(node_ids)
        self.k = k

    def neighbors(self, node_id: str) -> List[str]:
        n = len(self.node_ids)
        i = self._index[node_id]
        out = []
        for offset in range(1, self.k + 1):
            out.append(self.node_ids[(i + offset) % n])
            out.append(self.node_ids[(i - offset) % n])
        return list(dict.fromkeys(node for node in out if node != node_id))
