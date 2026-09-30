"""The neighbor view itself: the peers a node serves, and the peers it fetches from.

Construction from configuration lives in :mod:`appfl.decentralized.neighbor.resolve`; this
module is only the shape of the answer, so it depends on nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NeighborEndpoint:
    """One peer this node fetches from."""

    node_id: str
    #: Where to reach it. Required in peer-to-peer mode; unused in relay mode, where the
    #: relay is dialed instead and the peer is named rather than addressed.
    server_uri: str | None = None
    #: Mixing weight for this peer's model. `None` means uniform over the closed
    #: neighborhood, `1 / (|recv_from| + 1)`.
    weight: float | None = None


@dataclass
class Neighbors:
    """This node's view of the graph: who it serves, and who it fetches from."""

    #: Node ids permitted to request this node's model.
    send_to: list[str] = field(default_factory=list)
    #: Peers whose models this node fetches and averages.
    recv_from: list[NeighborEndpoint] = field(default_factory=list)

    @property
    def recv_from_ids(self) -> list[str]:
        return [neighbor.node_id for neighbor in self.recv_from]

    def endpoint(self, node_id: str) -> NeighborEndpoint | None:
        for neighbor in self.recv_from:
            if neighbor.node_id == node_id:
                return neighbor
        return None

    def require_undirected(self, local_id: str) -> None:
        """Raise unless this node serves exactly the peers it fetches from.

        Called only when the config did not declare `directed: true`. An empty `send_to` is
        unrestricted rather than asymmetric, so it is not checked -- that is the simulation
        default, where an access check between objects in one process would be theater.
        """
        if not self.send_to:
            return
        serving, fetching = set(self.send_to), set(self.recv_from_ids)
        if serving == fetching:
            return
        raise ValueError(
            f"neighbors for {local_id} describe a directed exchange: it serves "
            f"{sorted(serving)} but fetches from {sorted(fetching)}"
            + (
                f"; {sorted(serving - fetching)} would receive this node's model without ever "
                f"sending theirs back"
                if serving - fetching
                else ""
            )
            + (
                f"; {sorted(fetching - serving)} would be asked for a model this node is not "
                f"willing to reciprocate"
                if fetching - serving
                else ""
            )
            + ". Set `directed: true` if that asymmetry is deliberate -- a directed graph is "
            "supported, but it must be strongly connected and its consensus point is weighted "
            "by influence rather than being the plain average. If it was not deliberate, the "
            "two lists should name the same peers."
        )

    def may_serve(self, node_id: str) -> bool:
        """Whether ``node_id`` is allowed to request this node's model.

        An empty `send_to` means unrestricted. That is the right default for simulation, where
        everything runs in one process and an access check would only be theater; a deployment
        that cares states its list, and then the check is real.
        """
        return not self.send_to or str(node_id) in self.send_to

    def weights(self, local_id: str) -> dict[str, float]:
        """`{node_id: weight}` over this node and its `recv_from` peers, summing to one."""
        explicit = {n.node_id: n.weight for n in self.recv_from if n.weight is not None}
        if len(explicit) == len(self.recv_from):
            total = sum(explicit.values())
            if total > 1.0:
                raise ValueError(
                    f"neighbor weights for {local_id} sum to {total}, leaving no weight for "
                    f"the node's own model"
                )
            return {**explicit, local_id: 1.0 - total}
        if explicit:
            raise ValueError(
                f"{local_id}: some recv_from entries set `weight` and some do not. Set it on "
                f"all of them or none -- a partial assignment has no defensible completion."
            )
        uniform = 1.0 / (len(self.recv_from) + 1)
        return {n.node_id: uniform for n in self.recv_from} | {local_id: uniform}
