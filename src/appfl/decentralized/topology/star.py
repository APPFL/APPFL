"""One hub, N spokes: every node mixes with the hub and with nobody else."""

from __future__ import annotations

import math
from typing import Sequence

from appfl.decentralized.topology.base import Topology


class Star(Topology):
    """One hub, N spokes. Every spoke mixes with the hub; no two spokes mix with each other.

    The hub is a training node like any other -- it holds data, trains, and averages -- it just
    happens to be the only node with more than one neighbor.
    """

    def __init__(self, node_ids: Sequence[str], hub: str | None = None):
        super().__init__(node_ids)
        self.hub = hub if hub is not None else self.node_ids[0]
        if self.hub not in self._index:
            raise ValueError(f"hub {self.hub!r} is not one of {self.node_ids}")

    def neighbors(self, node_id: str) -> list[str]:
        if node_id == self.hub:
            return [n for n in self.node_ids if n != self.hub]
        return [self.hub]

    def layout(self) -> dict[str, tuple[float, float]]:
        """Hub at the center, spokes on a circle -- the shape the name describes."""
        spokes = [n for n in self.node_ids if n != self.hub]
        positions = {self.hub: (0.0, 0.0)}
        for i, node_id in enumerate(spokes):
            angle = 2 * math.pi * i / max(len(spokes), 1) - math.pi / 2
            positions[node_id] = (math.cos(angle), math.sin(angle))
        return positions
