"""The communication graph contract for decentralized federated learning.

Centralized FL has no graph over its participants: clients never exchange anything with one
another, and the aggregation point is not a participant that holds data or trains. In DFL the
graph is a first-class configuration object, because it is what defines the algorithm's
behavior -- who averages with whom, how fast information mixes, and whether the federation
converges at all.

A DFL graph may be **undirected or directed**, and must be connected -- strongly connected, if
directed. Connectivity is not a detail: a node that nothing can reach never influences anyone,
and a disconnected graph is two federations that will never agree. Every topology can report
its connectivity, and launchers check it before training starts rather than discovering the
problem in the accuracy curves.

The direction of an edge is *who receives whose model*. ``out_neighbors(i)`` are the nodes that
receive `i`'s model; ``in_neighbors(i)`` are the nodes whose models `i` receives and averages.
For an undirected graph the two coincide and :meth:`neighbors` says so.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Iterable, Sequence


class Topology(ABC):
    """A neighbor relation over DFL node IDs.

    Subclass and implement :meth:`neighbors` for an undirected graph; override
    :meth:`in_neighbors` and :meth:`out_neighbors` for a directed one. All of them return
    *peers only*, never the node itself, because that is what a communicator needs -- the
    node's own model is already local. Self-inclusion is handled by :meth:`mixing_weight`,
    which gives the weight a node puts on its own model.
    """

    #: Whether edges are one-way. Directed subclasses set this, and it changes which
    #: connectivity test applies.
    directed: bool = False

    def __init__(self, node_ids: Sequence[str]):
        self.node_ids = list(node_ids)
        self._index = {node_id: i for i, node_id in enumerate(self.node_ids)}

    @abstractmethod
    def neighbors(self, node_id: str) -> list[str]:
        """Peers that ``node_id`` exchanges models with. Excludes ``node_id`` itself.

        For a directed graph this is ambiguous, so directed subclasses override
        :meth:`in_neighbors` and :meth:`out_neighbors` and this returns the union -- everyone
        this node exchanges with in either direction.
        """

    def in_neighbors(self, node_id: str) -> list[str]:
        """Nodes whose models ``node_id`` receives and averages. Undirected: its neighbors."""
        return self.neighbors(node_id)

    def out_neighbors(self, node_id: str) -> list[str]:
        """Nodes that receive ``node_id``'s model. Undirected: its neighbors."""
        return self.neighbors(node_id)

    def degree(self, node_id: str) -> int:
        """In-degree: how many models this node averages. That is what weights normalize over."""
        return len(self.in_neighbors(node_id))

    def edges(self) -> Iterable[tuple[str, str]]:
        """Every ``(i, j)`` where ``j`` receives ``i``'s model.

        An undirected edge therefore appears twice, once in each direction, and a directed one
        appears once.
        """
        for node_id in self.node_ids:
            for neighbor in self.out_neighbors(node_id):
                yield (node_id, neighbor)

    def mixing_weight(self, i: str, j: str) -> float:
        """``pi_ij = 1 / (|in-neighbors of i| + 1)`` -- uniform over what ``i`` actually averages.

        This is the weight node ``i`` puts on node ``j``'s model. Normalizing over `i`'s
        *incoming* set is what keeps the mixing matrix row-stochastic, and therefore every
        update a convex combination, whether or not the graph is directed. Returns 0 when `j`
        does not send to `i`, so a model arriving by an unexpected path is ignored rather than
        silently over-weighted.

        On a fully connected graph this is ``1/n`` for every pair, which is what makes DFL
        reduce to equal-weight FedAvg.
        """
        if j != i and j not in self.in_neighbors(i):
            return 0.0
        return 1.0 / max(self.degree(i) + 1, 1)

    def is_connected(self) -> bool:
        """Whether information can flow from every node to every other node.

        Undirected graphs use the Fiedler value; directed graphs need *strong* connectivity,
        which is a strictly stronger condition. Symmetrizing a directed graph and testing that
        would pass `A -> B` with no return path -- a graph where `A` never learns anything from
        `B`, and where `B`'s model is the only one that matters.
        """
        if len(self.node_ids) <= 1:
            return True
        if self.directed:
            return self.is_strongly_connected()
        return self.fiedler_value() > 1e-9

    def is_strongly_connected(self) -> bool:
        """Whether every node reaches every other by following edge directions.

        Kosaraju's single-source test: a graph is strongly connected iff one node reaches all
        others in the graph and in its reverse.
        """
        if len(self.node_ids) <= 1:
            return True
        forward = {n: self.out_neighbors(n) for n in self.node_ids}
        backward: dict[str, list[str]] = {n: [] for n in self.node_ids}
        for node, targets in forward.items():
            for target in targets:
                backward[target].append(node)

        def reaches_all(adjacency) -> bool:
            seen, stack = set(), [self.node_ids[0]]
            while stack:
                node = stack.pop()
                if node in seen:
                    continue
                seen.add(node)
                stack.extend(adjacency[node])
            return len(seen) == len(self.node_ids)

        return reaches_all(forward) and reaches_all(backward)

    def fiedler_value(self) -> float:
        """``lambda_2(L(G))``, algebraic connectivity. Zero if and only if disconnected.

        Higher means faster mixing: a complete graph on ``n`` nodes gives ``n``, a ring gives
        roughly ``2(1 - cos(2*pi/n))``, which goes to zero as the ring grows. Worth logging
        with every run, since it is the single number that explains why one topology reaches
        consensus in ten rounds and another takes hundreds.

        Defined for undirected graphs. On a directed one the adjacency is symmetrized first, so
        the result is a proxy for the underlying communication structure and **not** a
        connectivity test -- use :meth:`is_strongly_connected` for that.
        """
        import numpy as np

        n = len(self.node_ids)
        adjacency = np.zeros((n, n))
        for i, j in self.edges():
            adjacency[self._index[i], self._index[j]] = 1.0
            adjacency[self._index[j], self._index[i]] = 1.0  # assumed undirected
        laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
        eigenvalues = np.sort(np.linalg.eigvalsh(laplacian))
        return float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0

    # -- inspection ---------------------------------------------------------------------

    def layout(self) -> dict[str, tuple[float, float]]:
        """Node positions for drawing, as ``{node_id: (x, y)}``.

        Evenly spaced on a circle, which reads well for the graphs where position carries no
        meaning. Subclasses override when they have something better to say: a star puts its
        hub in the middle, and a random geometric graph knows where its nodes actually are.
        """
        n = len(self.node_ids)
        if n == 1:
            return {self.node_ids[0]: (0.0, 0.0)}
        return {
            node_id: (
                math.cos(2 * math.pi * i / n - math.pi / 2),
                math.sin(2 * math.pi * i / n - math.pi / 2),
            )
            for i, node_id in enumerate(self.node_ids)
        }

    def draw(
        self,
        path: str | None = None,
        ax: Any | None = None,
        title: str | None = None,
        with_labels: bool = True,
        figsize: tuple[float, float] = (5.5, 5.5),
        node_color: str = "#156082",
        edge_color: str = "#8a8a8a",
        fontsize: float = 7.0,
    ) -> Any:
        """Draw the graph and return the axes; save to ``path`` if given.

        A topology is the one part of a DFL configuration that is genuinely hard to read as
        text -- ``send_to``/``recv_from`` lists over sixteen nodes are technically complete and
        practically opaque. Directed graphs are drawn with arrowheads pointing from sender to
        receiver, so a missing return path is visible rather than inferred.

        :param path: file to save to. The extension picks the format, as usual for matplotlib.
        :param ax: draw onto existing axes instead of making a figure, for putting several
            topologies side by side.
        :param title: defaults to the class name plus :meth:`describe`.
        """
        import matplotlib.pyplot as plt

        positions = self.layout()
        created = ax is None
        if created:
            _, ax = plt.subplots(figsize=figsize)

        # Size the markers to the longest label rather than fixing them, so ids longer than
        # "Node7" -- a real deployment names sites, not indices -- are not clipped. Widths are
        # in points, which is also what arrow shrink is measured in, so the two stay consistent
        # at any figure size.
        longest = max((len(str(n)) for n in self.node_ids), default=1)
        diameter = max(18.0, longest * fontsize * 0.62 + 7.0)
        radius = diameter / 2.0

        drawn = set()
        for sender, receiver in self.edges():
            if not self.directed:
                # One line per undirected edge; `edges()` yields both directions.
                key = frozenset((sender, receiver))
                if key in drawn:
                    continue
                drawn.add(key)
            ax.annotate(
                "",
                xy=positions[receiver],
                xytext=positions[sender],
                arrowprops={
                    "arrowstyle": "-|>" if self.directed else "-",
                    "color": edge_color,
                    "linewidth": 1.1,
                    # In points, so the arrowheads clear the node markers whatever the scale.
                    "shrinkA": radius + 1.5,
                    "shrinkB": radius + 1.5,
                },
            )

        xs = [positions[n][0] for n in self.node_ids]
        ys = [positions[n][1] for n in self.node_ids]
        ax.scatter(
            xs,
            ys,
            s=diameter**2,
            c=node_color,
            zorder=3,
            edgecolors="white",
            linewidths=1.5,
        )
        if with_labels:
            for node_id in self.node_ids:
                ax.annotate(
                    node_id,
                    positions[node_id],
                    color="white",
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    zorder=4,
                )

        summary = self.describe()
        if title is None:
            connectivity = (
                f"strongly connected={summary['strongly_connected']}"
                if self.directed
                else f"fiedler={summary['fiedler_value']:.3f}"
            )
            title = (
                f"{type(self).__name__}  ({summary['n_nodes']} nodes, "
                f"{summary['n_edges']} edges, {connectivity})"
            )
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal")
        ax.margins(0.18)
        ax.axis("off")

        if path is not None:
            ax.get_figure().savefig(path, dpi=150, bbox_inches="tight")
            if created:
                plt.close(ax.get_figure())
        return ax

    def to_ascii(self, max_nodes: int = 24) -> str:
        """The adjacency matrix as text: row = sender, column = receiver.

        Needed because the matplotlib path is useless in the place a topology most often has to
        be checked -- a batch job's log on a machine with no display. Beyond ``max_nodes`` the
        matrix stops being readable, so only the summary is returned.
        """
        summary = ", ".join(f"{k}={v}" for k, v in self.describe().items())
        header = f"{type(self).__name__}: {summary}"
        if len(self.node_ids) > max_nodes:
            return f"{header}\n(matrix omitted above {max_nodes} nodes)"

        width = max(len(n) for n in self.node_ids)
        digits = len(str(len(self.node_ids) - 1))
        lines = [
            header,
            "row sends to column"
            + (" (undirected, so symmetric)" if not self.directed else ""),
            " " * (width + 2)
            + " ".join(f"{i:>{digits}}" for i in range(len(self.node_ids))),
        ]
        for i, sender in enumerate(self.node_ids):
            out = set(self.out_neighbors(sender))
            row = " ".join(
                f"{('.' if receiver == sender else ('1' if receiver in out else '0')):>{digits}}"
                for receiver in self.node_ids
            )
            lines.append(f"{sender:>{width}} {i:>{digits}} {row}")
        return "\n".join(lines)

    def describe(self) -> dict[str, float]:
        """Summary logged alongside every run, so topologies are comparable after the fact."""
        degrees = [self.degree(node_id) for node_id in self.node_ids]
        summary = {
            "n_nodes": len(self.node_ids),
            "n_edges": sum(1 for _ in self.edges()) // (1 if self.directed else 2),
            "mean_in_degree": sum(degrees) / max(1, len(degrees)),
        }
        if self.directed:
            summary["strongly_connected"] = self.is_strongly_connected()
        else:
            summary["fiedler_value"] = self.fiedler_value()
        return summary
