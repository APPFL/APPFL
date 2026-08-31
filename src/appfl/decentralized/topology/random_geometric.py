"""Agents in the unit square, edges within a radius."""

from __future__ import annotations

import math

from typing import List, Optional, Sequence

from appfl.decentralized.topology.base import Topology


class RandomGeometric(Topology):
    """Agents placed uniformly in the unit square; edges within ``radius``.

    The v2 topology-ablation arm. Unlike a ring or a complete graph it is *irregular*, which
    is the interesting case: degrees differ, so the choice between uniform closed-neighborhood
    weights and Metropolis-Hastings stops being cosmetic.

    A raw radius graph is frequently disconnected, and ADKO's guarantees assume connectivity,
    so the closest pair of components is bridged repeatedly until the graph is connected --
    preserving the geometric construction while guaranteeing every reported run is valid.
    Same approach as the reference.
    """

    def __init__(self, agent_ids: Sequence[str], radius: float = 0.5, seed: int = 0):
        super().__init__(agent_ids)
        self.radius = radius
        self.seed = seed
        self._adjacency = self._build()

    def _build(self):
        import numpy as np

        n = len(self.agent_ids)
        rng = np.random.RandomState(self.seed)
        positions = rng.uniform(0.0, 1.0, size=(n, 2))
        deltas = positions[:, None, :] - positions[None, :, :]
        distances = np.sqrt((deltas**2).sum(axis=-1))
        adjacency = (distances <= self.radius) & ~np.eye(n, dtype=bool)

        # Bridge components until connected, cheapest edge first.
        while True:
            components = self._components(adjacency, n)
            if len(components) <= 1:
                break
            first = components[0]
            rest = [i for c in components[1:] for i in c]
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
                stack.extend(k for k in range(n) if adjacency[node, k] and k not in component)
            components.append(sorted(component))
        return components

    def neighbors(self, agent_id: str) -> List[str]:
        i = self._index[agent_id]
        return [
            self.agent_ids[j]
            for j in range(len(self.agent_ids))
            if self._adjacency[i, j]
        ]
