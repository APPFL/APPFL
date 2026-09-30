"""Everyone exchanges with everyone."""

from __future__ import annotations

from typing import List

from appfl.decentralized.topology.base import Topology


class FullyConnected(Topology):
    """Every node is every other node's neighbor.

    The reference point for DFL: each node averages all ``n`` models every round, so all
    nodes hold the same model afterwards and the run is identical to synchronous FedAvg with
    equal client weights. Also the most expensive graph -- ``n(n-1)/2`` edges -- which is why
    it is a baseline rather than a deployment.
    """

    def neighbors(self, node_id: str) -> List[str]:
        return [n for n in self.node_ids if n != node_id]
