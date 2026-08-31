"""Communication graphs -- who may send to whom, and with what weight.

One file per graph, mirroring how APPFL organizes aggregators, because a topology is exactly
the kind of thing a user adds: subclass :class:`Topology`, implement ``neighbors``, register
it below or construct it directly.

``build_topology`` resolves by name so the graph is a YAML knob like everything else::

    build_topology("ring", agent_ids, k=2)
    build_topology("random_geometric", agent_ids, radius=0.4, seed=7)
"""

from typing import Sequence

from appfl.decentralized.topology.base import Topology
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
    "build_topology",
]

_REGISTRY = {
    "fully_connected": FullyConnected,
    "complete": FullyConnected,  # the ADKO reference's name for the same graph
    "ring": Ring,
    "star": Star,
    "random_geometric": RandomGeometric,
}


def build_topology(name: str, agent_ids: Sequence[str], **kwargs) -> Topology:
    """Config-driven construction, so topology is a YAML knob like everything else in APPFL."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown topology '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](agent_ids, **kwargs)
