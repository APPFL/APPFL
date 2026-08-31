"""One hub, N spokes -- the shape APPFL uses today."""

from __future__ import annotations


from typing import List, Optional, Sequence

from appfl.decentralized.topology.base import Topology


class Star(Topology):
    """One hub, N spokes. What APPFL does today; kept as the migration reference point."""

    def __init__(self, agent_ids: Sequence[str], hub: Optional[str] = None):
        super().__init__(agent_ids)
        self.hub = hub if hub is not None else self.agent_ids[0]

    def neighbors(self, agent_id: str) -> List[str]:
        if agent_id == self.hub:
            return [a for a in self.agent_ids if a != self.hub]
        return [self.hub]
