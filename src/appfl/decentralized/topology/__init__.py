"""Communication graphs for decentralized federated learning -- who exchanges with whom.

One file per graph, mirroring how APPFL organizes aggregators and trainers, because a topology
is exactly the kind of thing a user adds: subclass :class:`Topology`, implement ``neighbors``,
and either register it here or pass the instance directly.

A graph is not something a decentralized *node* holds -- a node knows only its own neighbors,
via :mod:`appfl.decentralized.neighbor`. A topology is how those neighbor lists get derived
when something legitimately owns the whole federation: a simulation launcher, or a node config
in ``from_topo`` mode, where the ids are generated and every node therefore derives the same
graph::

    neighbors:
      mode: "from_topo"
      topology: "ring"
      topology_kwargs:
        k: 2
      num_nodes: 16

In code::

    from appfl.decentralized.topology import build_topology

    topology = build_topology("fully_connected", ["Node0", "Node1", "Node2"])
    topology.neighbors("Node0")      # ['Node1', 'Node2']
    topology.mixing_weight("Node0", "Node1")   # 1/3
    topology.describe()              # n_nodes, n_edges, mean_degree, fiedler_value
"""

from typing import Sequence

from appfl.decentralized.topology.base import Topology
from appfl.decentralized.topology.custom import Custom
from appfl.decentralized.topology.fully_connected import FullyConnected
from appfl.decentralized.topology.random_geometric import RandomGeometric
from appfl.decentralized.topology.ring import Ring
from appfl.decentralized.topology.star import Star

__all__ = [
    "Topology",
    "FullyConnected",
    "Ring",
    "Star",
    "RandomGeometric",
    "Custom",
    "build_topology",
]

_REGISTRY = {
    "fully_connected": FullyConnected,
    "complete": FullyConnected,  # the same graph, under the name the DBO literature uses
    "ring": Ring,
    "star": Star,
    "random_geometric": RandomGeometric,
    "custom": Custom,
}


def build_topology(name: str, node_ids: Sequence[str], **kwargs) -> Topology:
    """Resolve a topology by name, so the graph is configuration rather than code."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown topology '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](node_ids, **kwargs)
