"""A random ``k``-regular graph: the sparse topology a large federation actually runs on.

``random_geometric`` is sparse but its degree varies with position, so a result there mixes the
method's behaviour with the degree distribution. A ``k``-regular graph fixes the degree at
``k`` for every node and leaves only the structure random, which is what makes "the method
survives at degree 4 with a hundred agents" a statement about the method.

Built with the pairing model and retried until the draw is simple (no self-loops, no repeated
edges); if the retries run out, the partial graph is completed greedily and then bridged, so a
topology is always returned and always connected. Connectivity matters here more than in
centralized FL: a claim or a verdict that cannot reach a component is knowledge the federation
paid for and did not get.
"""

from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from appfl.decentralized.topology.base import Topology


class RandomRegular(Topology):
    """Every node has exactly ``k`` neighbors (``k+1`` is impossible for odd ``n*k``).

    Args:
        node_ids: the federation's node ids.
        k: degree. Clipped to ``n - 1``; ``n * k`` must be even, so ``k`` is reduced by one
            when both are odd.
        seed: draw seed. The same ``(node_ids, k, seed)`` always gives the same graph, which is
            what lets an arm comparison hold the graph fixed across seeds of the method.
        tries: pairing-model attempts before falling back to the greedy completion.
    """

    def __init__(self, node_ids: Sequence[str], k: int = 4, seed: int = 0, tries: int = 50):
        super().__init__(node_ids)
        n = len(self.node_ids)
        k = max(1, min(int(k), n - 1)) if n > 1 else 0
        if (n * k) % 2 == 1:
            k -= 1
        self.k, self.seed = k, int(seed)
        self._adj = self._build(n, k, int(seed), int(tries))

    def _build(self, n: int, k: int, seed: int, tries: int) -> Dict[int, Set[int]]:
        rng = np.random.default_rng([seed, n, k])
        adj: Dict[int, Set[int]] = {i: set() for i in range(n)}
        if n < 2 or k < 1:
            return adj
        for _ in range(tries):
            adj = {i: set() for i in range(n)}
            stubs = rng.permutation(np.repeat(np.arange(n), k))
            ok = True
            for a, b in zip(stubs[0::2], stubs[1::2]):
                a, b = int(a), int(b)
                if a == b or b in adj[a]:
                    ok = False
                    break
                adj[a].add(b)
                adj[b].add(a)
            if ok:
                return self._bridge(adj, rng)
        for i in range(n):  # greedy completion of whatever the last draw managed
            while len(adj[i]) < k:
                free = [j for j in range(n) if j != i and len(adj[j]) < k and j not in adj[i]]
                if not free:
                    break
                j = int(rng.choice(free))
                adj[i].add(j)
                adj[j].add(i)
        return self._bridge(adj, rng)

    @staticmethod
    def _bridge(adj: Dict[int, Set[int]], rng: np.random.Generator) -> Dict[int, Set[int]]:
        """Join the components, one edge each, so every node can reach every other."""
        comps = RandomRegular._components(adj)
        for a, b in zip(comps, comps[1:]):
            i, j = int(rng.choice(sorted(a))), int(rng.choice(sorted(b)))
            adj[i].add(j)
            adj[j].add(i)
        return adj

    @staticmethod
    def _components(adj: Dict[int, Set[int]]) -> List[Set[int]]:
        seen: Set[int] = set()
        out: List[Set[int]] = []
        for start in adj:
            if start in seen:
                continue
            stack, comp = [start], set()
            while stack:
                v = stack.pop()
                if v in comp:
                    continue
                comp.add(v)
                stack.extend(adj[v] - comp)
            seen |= comp
            out.append(comp)
        return out

    def neighbors(self, node_id: str) -> List[str]:
        i = self.node_ids.index(node_id)
        return [self.node_ids[j] for j in sorted(self._adj[i])]
