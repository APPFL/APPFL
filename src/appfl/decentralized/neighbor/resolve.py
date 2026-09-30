"""Turn a ``neighbors`` config block into a concrete
:class:`~appfl.decentralized.neighbor.Neighbors`.

Two modes, and the difference between them is who is entitled to know the whole federation.
See the package docstring for what each is for.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from appfl.decentralized.neighbor.base import NeighborEndpoint, Neighbors
from appfl.decentralized.topology import Topology, build_topology


#: Derive this node's neighbors from a named graph over generated node ids. Simulation only.
FROM_TOPOLOGY = "from_topo"
#: Name this node's neighbors directly. The only mode a real deployment can use.
EXPLICIT = "explicit"
NEIGHBOR_MODES = (FROM_TOPOLOGY, EXPLICIT)

#: Node ids `from_topo` generates, and which the launchers assign to match.
def generated_node_ids(num_nodes: int) -> List[str]:
    """`Node0 .. Node{num_nodes-1}` -- the roster `from_topo` assumes."""
    return [f"Node{i}" for i in range(int(num_nodes))]


def build_topology_from_config(config: Any) -> Topology:
    """Build the graph a `from_topo` neighbors block names, over the generated node ids.

    Launchers call this once and pass the result to :func:`resolve_neighbors`, so a thousand
    nodes do not each construct their own copy of the same graph.

    Only a `from_topo` block can be built from: an `explicit` one names its neighbors and says
    nothing about a graph. Rejecting it here rather than in each launcher keeps the check in
    the one place the requirement exists -- and the mistake it catches is passing a deployment
    config to a simulation launcher, which would otherwise surface as a missing `num_nodes`.
    """
    mode = config.get("mode", None)
    if mode != FROM_TOPOLOGY:
        raise ValueError(
            f"cannot build a topology from a neighbors block with mode {mode!r}; only "
            f"{FROM_TOPOLOGY!r} names a graph. A block using 'explicit' describes one site of "
            f"a real deployment, where no global graph exists -- run it with a launcher that "
            f"takes one node per process rather than a simulation launcher."
        )
    return build_topology(
        str(config.get("topology", "fully_connected")),
        generated_node_ids(config["num_nodes"]),
        **(_as_dict(config.get("topology_kwargs", None))),
    )


def _as_dict(node: Any) -> Dict[str, Any]:
    """A plain dict from a config node that may be absent or an empty (falsy) DictConfig."""
    if node is None:
        return {}
    return {key: node[key] for key in node} if len(node) else {}


def resolve_neighbors(
    config: Any,
    node_id: str,
    topology: Optional[Topology] = None,
) -> Neighbors:
    """Turn a `neighbors` config block into a concrete :class:`Neighbors`.

    :param config: the block. `mode: from_topo` derives the lists from a named graph;
        `mode: explicit` reads `send_to` / `recv_from` directly.
    :param node_id: the node this is being resolved for.
    :param topology: an already-built graph for `from_topo`, so a launcher can construct it
        once for the whole federation instead of once per node. Built from the block when
        omitted.
    """
    if config is None:
        return Neighbors()

    mode = config.get("mode", None)
    if mode is None:
        raise ValueError(
            f"neighbors block for {node_id} has no `mode`. Set one of {list(NEIGHBOR_MODES)}: "
            f"`{FROM_TOPOLOGY}` derives neighbors from a named graph and is for simulation "
            f"only; `{EXPLICIT}` names them directly and is what a deployed node uses."
        )
    if mode not in NEIGHBOR_MODES:
        raise ValueError(
            f"neighbors mode {mode!r} is not understood; expected one of {list(NEIGHBOR_MODES)}"
        )

    if mode == FROM_TOPOLOGY:
        topology = topology if topology is not None else build_topology_from_config(config)
        if node_id not in topology.node_ids:
            raise ValueError(
                f"node_id {node_id!r} is not among the ids `{FROM_TOPOLOGY}` generates "
                f"({topology.node_ids[:4]}{'...' if len(topology.node_ids) > 4 else ''}). "
                f"This mode assumes generated ids, so the launcher must assign them to match."
            )
        return Neighbors(
            send_to=list(topology.out_neighbors(node_id)),
            recv_from=[
                NeighborEndpoint(node_id=peer) for peer in topology.in_neighbors(node_id)
            ],
        )

    send_to = [str(n) for n in (config.get("send_to", []) or [])]
    recv_from = []
    for entry in config.get("recv_from", []) or []:
        if isinstance(entry, str):  # bare id, for relay mode where no endpoint is needed
            recv_from.append(NeighborEndpoint(node_id=entry))
            continue
        recv_from.append(
            NeighborEndpoint(
                node_id=str(entry["node_id"]),
                server_uri=entry.get("server_uri", None),
                weight=entry.get("weight", None),
            )
        )
    neighbors = Neighbors(send_to=send_to, recv_from=recv_from)
    if not bool(config.get("directed", False)):
        neighbors.require_undirected(node_id)
    return neighbors


def resolve_all_neighbors_from_topology(
    topology: Topology, node_ids: Optional[Sequence[str]] = None
) -> Dict[str, Neighbors]:
    """Every node's neighbor view, keyed by node id, derived from one graph.

    The counterpart of :func:`resolve_neighbors`, which answers for a single node from its own
    config. This answers for the whole federation at once, which only a simulation launcher is
    in a position to ask -- and is why it takes a graph rather than a config block.
    """
    return {
        node_id: Neighbors(
            # out-neighbors receive this node's model, so they are who it must serve;
            # in-neighbors are whose models it fetches. Equal unless the graph is directed.
            send_to=list(topology.out_neighbors(node_id)),
            recv_from=[
                NeighborEndpoint(node_id=peer) for peer in topology.in_neighbors(node_id)
            ],
        )
        for node_id in (node_ids if node_ids is not None else topology.node_ids)
    }
