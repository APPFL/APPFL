"""Everyone hears everyone."""

from __future__ import annotations


from typing import List, Optional, Sequence

from appfl.decentralized.topology.base import Topology


class FullyConnected(Topology):
    """Everyone hears everyone. The ADKO reference default."""

    def neighbors(self, agent_id: str) -> List[str]:
        return [a for a in self.agent_ids if a != agent_id]
