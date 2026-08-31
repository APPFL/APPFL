"""A cycle: each agent talks to k neighbours either side."""

from __future__ import annotations

import math

from typing import List, Optional, Sequence

from appfl.decentralized.topology.base import Topology


class Ring(Topology):
    """Each agent talks to ``k`` neighbors either side. Cheap, slow-mixing, low connectivity.

    The stress case for the coordination study: information needs O(n/k) hops to cross the
    federation, so the recency term in fidelity-aware pruning bites before a finding arrives.
    """

    def __init__(self, agent_ids: Sequence[str], k: int = 1):
        super().__init__(agent_ids)
        self.k = k

    def neighbors(self, agent_id: str) -> List[str]:
        n = len(self.agent_ids)
        i = self._index[agent_id]
        out = []
        for offset in range(1, self.k + 1):
            out.append(self.agent_ids[(i + offset) % n])
            out.append(self.agent_ids[(i - offset) % n])
        return list(dict.fromkeys(a for a in out if a != agent_id))
