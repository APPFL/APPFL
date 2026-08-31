"""The communication graph contract, and the weights ADKO reads off it.

ADKO assumes ``G`` undirected and connected -- its algebraic connectivity ``lambda_2(L(G))``
must be positive, and convergence speed depends on it. The reference defaults to a complete
graph and logs the Fiedler value for every run.

Topology is also the natural independent variable for a scaling study: sweep the graph, watch
result quality against bits sent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Sequence, Tuple


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
