"""A graph given explicitly, as an edge list or an adjacency map."""

from __future__ import annotations

import warnings
from typing import Iterable, Sequence

from appfl.decentralized.topology.base import Topology


class Custom(Topology):
    """An arbitrary graph, written out rather than generated.

    Real federations rarely match a named family: peering follows institutional agreements,
    firewall rules and network locality. This is the escape hatch, and it is the only topology
    whose YAML fully describes the graph::

        topology: "custom"
        topology_kwargs:
          edges:
            - ["Node0", "Node1"]
            - ["Node1", "Node2"]

    or equivalently::

        topology_kwargs:
          adjacency:
            Node0: ["Node1"]
            Node1: ["Node0", "Node2"]
            Node2: ["Node1"]

    An edge reads ``[sender, receiver]``: ``["Node0", "Node1"]`` means Node1 receives Node0's
    model. With ``directed: false`` (the default) each edge is symmetrized, so that is also an
    exchange in both directions; with ``directed: true`` it is one-way, and Node0 never sees
    Node1's model unless the reverse edge is given too::

        topology: "custom"
        topology_kwargs:
          directed: true
          edges:
            - ["Node0", "Node1"]      # Node1 averages Node0's model
            - ["Node1", "Node2"]
            - ["Node2", "Node0"]      # closes the cycle, so the graph is strongly connected

    Undirected is the default because a one-directional entry is far more often an omission
    than an intent, and because uniform weights on an undirected graph are the case that
    reaches the plain average -- the case that reproduces centralized FedAvg. When the
    symmetrization actually changes the input, it is reported through :mod:`warnings` rather
    than done silently, so a config that was not taken literally says so. Pass
    ``warn_on_repair=False`` to suppress that.

    A directed graph must be **strongly connected**, which is stricter than it looks: a chain
    with no return path leaves the first node influencing everyone and learning from no one.
    :meth:`~appfl.decentralized.topology.Topology.is_connected` applies the right test; the
    launchers warn rather than refuse, since running a disconnected graph is a legitimate
    baseline.
    """

    def __init__(
        self,
        node_ids: Sequence[str],
        edges: Iterable[tuple[str, str]] | None = None,
        adjacency: dict[str, Sequence[str]] | None = None,
        directed: bool = False,
        warn_on_repair: bool = True,
    ):
        super().__init__(node_ids)
        self.directed = directed
        if (edges is None) == (adjacency is None):
            raise ValueError(
                "Custom topology needs exactly one of `edges` or `adjacency`"
            )

        self._out: dict[str, list[str]] = {n: [] for n in self.node_ids}
        self._in: dict[str, list[str]] = {n: [] for n in self.node_ids}
        pairs = (
            [(str(a), str(b)) for a, b in edges]
            if edges is not None
            else [
                (str(node), str(neighbor))
                for node, neighbors in adjacency.items()
                for neighbor in neighbors
            ]
        )
        given = set()
        for sender, receiver in pairs:
            for name in (sender, receiver):
                if name not in self._out:
                    raise ValueError(
                        f"edge references unknown node {name!r}; nodes are {self.node_ids}"
                    )
            if sender == receiver:
                raise ValueError(f"node {sender!r} cannot be its own neighbor")
            given.add((sender, receiver))
            self._add(sender, receiver)
            if not directed:
                self._add(receiver, sender)

        repaired = sorted((b, a) for a, b in given if (b, a) not in given)
        if repaired and not directed and warn_on_repair:
            warnings.warn(
                f"Custom topology: {len(repaired)} edge(s) were given in one direction only "
                f"and have been symmetrized, because the graph was not declared directed: "
                f"{repaired[:5]}{'...' if len(repaired) > 5 else ''}. Pass directed=True to "
                f"keep them one-way, or warn_on_repair=False to silence this.",
                UserWarning,
                stacklevel=3,
            )

    def _add(self, sender: str, receiver: str) -> None:
        if receiver not in self._out[sender]:
            self._out[sender].append(receiver)
        if sender not in self._in[receiver]:
            self._in[receiver].append(sender)

    def neighbors(self, node_id: str) -> list[str]:
        """Everyone this node exchanges with, in either direction."""
        return list(dict.fromkeys(self._out[node_id] + self._in[node_id]))

    def in_neighbors(self, node_id: str) -> list[str]:
        return list(self._in[node_id])

    def out_neighbors(self, node_id: str) -> list[str]:
        return list(self._out[node_id])
