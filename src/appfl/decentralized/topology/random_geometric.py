"""Nodes placed in the unit square, edges within a radius."""

from __future__ import annotations

from typing import Sequence

from appfl.decentralized.topology.base import Topology


class RandomGeometric(Topology):
    """Nodes placed uniformly in the unit square; an edge wherever they are within ``radius``.

    The interesting case for weighting, because unlike a ring or a complete graph it is
    *irregular*: degrees differ, so the choice between uniform closed-neighborhood weights and
    Metropolis-Hastings stops being cosmetic. It is also a reasonable stand-in for a real
    federation where peering follows geography or network locality.

    A raw radius graph is frequently disconnected, so the closest pair of components is
    bridged repeatedly until it is connected -- preserving the geometric construction while
    guaranteeing the run is valid.
    """

    def __init__(self, node_ids: Sequence[str], radius: float = 0.5, seed: int = 0):
        super().__init__(node_ids)
        self.radius = radius
        self.seed = seed
        self._adjacency = self._build()

    def _build(self):
        import numpy as np

        n = len(self.node_ids)
        rng = np.random.RandomState(self.seed)
        positions = rng.uniform(0.0, 1.0, size=(n, 2))
        # Kept so `layout()` can draw the graph where the nodes actually are, rather than
        # rearranging them onto a circle and discarding the geometry that defined the edges.
        self.positions = positions
        deltas = positions[:, None, :] - positions[None, :, :]
        distances = np.sqrt((deltas**2).sum(axis=-1))
        adjacency = (distances <= self.radius) & ~np.eye(n, dtype=bool)

        # Bridge components until connected, cheapest edge first.
        while True:
            components = self._components(adjacency, n)
            if len(components) <= 1:
                break
            first = components[0]
            rest = [i for component in components[1:] for i in component]
            sub = distances[np.ix_(first, rest)]
            a, b = np.unravel_index(np.argmin(sub), sub.shape)
            i, j = first[a], rest[b]
            adjacency[i, j] = adjacency[j, i] = True
        return adjacency

    @staticmethod
    def _components(adjacency, n):
        unseen = set(range(n))
        components = []
        while unseen:
            stack = [min(unseen)]
            component = set()
            while stack:
                node = stack.pop()
                if node in component:
                    continue
                component.add(node)
                unseen.discard(node)
                stack.extend(
                    k for k in range(n) if adjacency[node, k] and k not in component
                )
            components.append(sorted(component))
        return components

    def layout(self) -> dict[str, tuple[float, float]]:
        """The positions the edges were derived from."""
        return {
            node_id: (float(self.positions[i][0]), float(self.positions[i][1]))
            for i, node_id in enumerate(self.node_ids)
        }

    def neighbors(self, node_id: str) -> list[str]:
        i = self._index[node_id]
        return [
            self.node_ids[j] for j in range(len(self.node_ids)) if self._adjacency[i, j]
        ]
