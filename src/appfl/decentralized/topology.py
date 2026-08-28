"""The communication graph ``G = (V, E)`` and its mixing weights.

ADKO assumes ``G`` undirected and connected -- its algebraic connectivity ``lambda_2(L(G))``
must be positive, and convergence speed depends on it. The reference implementation defaults
to a complete graph (``make_complete_graph``) and logs the Fiedler value for every run.

Topology is also the natural independent variable for a scaling study: sweep the graph, watch
result quality against bits sent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


class Topology(ABC):
    """A neighbor relation over agent IDs.

    ADKO's neighborhood ``N_i`` includes the agent itself, but :meth:`neighbors` returns
    *peers only* -- that is what the transport needs. Self-inclusion is handled where it
    matters: :meth:`uniform_weight` gives ``pi_ii`` for ``i == j``, so an agent's own tokens
    carry their proper weight in the reasoning score without ever being sent anywhere.
    """

    def __init__(self, agent_ids: Sequence[str]):
        self.agent_ids = list(agent_ids)
        self._index = {a: i for i, a in enumerate(self.agent_ids)}

    @abstractmethod
    def neighbors(self, agent_id: str) -> List[str]:
        """Peers that ``agent_id`` may send to. Excludes ``agent_id`` itself."""

    def degree(self, agent_id: str) -> int:
        return len(self.neighbors(agent_id))

    def edges(self) -> Iterable[Tuple[str, str]]:
        for a in self.agent_ids:
            for b in self.neighbors(a):
                yield (a, b)

    def uniform_weight(self, i: str, j: str) -> float:
        """``pi_ij = 1 / (|N_i| + 1)`` -- uniform over the closed neighborhood including self.

        This is what the ADKO reference implementation uses
        (``run_suzuki.py::Agent._peer_G_and_Lambda``). On a regular graph it coincides
        exactly with Metropolis-Hastings weights; it diverges only when degrees differ.
        Returns 0 for non-neighbors, so a token arriving by an unexpected path is ignored
        rather than silently over-weighted.
        """
        if j != i and j not in self.neighbors(i):
            return 0.0
        return 1.0 / max(self.degree(i) + 1, 1)

    def fiedler_value(self) -> float:
        """``lambda_2(L(G))``, algebraic connectivity. Zero iff the graph is disconnected.

        ADKO assumes this is positive; a run whose topology reports 0 violates the paper's
        standing assumption and its guarantees do not apply. Worth asserting in an
        experiment driver rather than discovering it in the result curves. The reference
        logs it for every run.
        """
        import numpy as np

        n = len(self.agent_ids)
        adjacency = np.zeros((n, n))
        for a, b in self.edges():
            adjacency[self._index[a], self._index[b]] = 1.0
            adjacency[self._index[b], self._index[a]] = 1.0  # ADKO assumes G undirected
        laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
        eigenvalues = np.sort(np.linalg.eigvalsh(laplacian))
        return float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0

    def describe(self) -> Dict[str, float]:
        """Summary logged alongside every run so topologies are comparable after the fact."""
        degrees = [self.degree(a) for a in self.agent_ids]
        return {
            "n_agents": len(self.agent_ids),
            "mean_degree": sum(degrees) / max(1, len(degrees)),
            "fiedler_value": self.fiedler_value(),
        }


class FullyConnected(Topology):
    """Everyone hears everyone. The ADKO reference default."""

    def neighbors(self, agent_id: str) -> List[str]:
        return [a for a in self.agent_ids if a != agent_id]


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


class Star(Topology):
    """One hub, N spokes. What APPFL does today; kept as the migration reference point."""

    def __init__(self, agent_ids: Sequence[str], hub: Optional[str] = None):
        super().__init__(agent_ids)
        self.hub = hub if hub is not None else self.agent_ids[0]

    def neighbors(self, agent_id: str) -> List[str]:
        if agent_id == self.hub:
            return [a for a in self.agent_ids if a != self.hub]
        return [self.hub]


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


def build_topology(name: str, agent_ids: Sequence[str], **kwargs) -> Topology:
    """Config-driven construction, so topology is a YAML knob like everything else in APPFL."""
    registry = {
        "fully_connected": FullyConnected,
        "complete": FullyConnected,  # the reference's name for the same graph
        "ring": Ring,
        "star": Star,
        "random_geometric": RandomGeometric,
    }
    if name not in registry:
        raise ValueError(f"Unknown topology '{name}'. Available: {sorted(registry)}")
    return registry[name](agent_ids, **kwargs)
