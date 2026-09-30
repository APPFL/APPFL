"""Who a node exchanges models with -- the only thing about the graph a node needs to know.

A decentralized node's runtime contract is **local**. It knows the peers it fetches from and
the peers it is willing to serve, and nothing else: not the roster, not the endpoints of nodes
it never talks to, not the shape of the federation. A deployed site should not have to hold a
byte-identical copy of a global graph, and should not learn the membership of a federation it
participates in only at the edges.

So the graph is expressed once per node, from that node's point of view::

    neighbors:
      mode: "explicit"
      send_to: ["Node1", "Node2"]         # who may request MY model
      recv_from:                          # whose models I fetch and average
        - node_id: "Node1"
          server_uri: "site-b.example.org:50051"
        - node_id: "Node2"
          server_uri: "site-c.example.org:50051"

The two lists are not duplicates of one edge; they do different jobs:

* ``recv_from`` drives this node's requests and its aggregation. It needs an endpoint per peer
  in peer-to-peer mode, and none in relay mode, where the relay is the only endpoint.
* ``send_to`` is an access-control declaration *about this node's own model*. A peer-to-peer
  servicer refuses a request from anyone not on it, and a relay assembles its permission map
  out of what each node declares on check-in -- which is how a relay can reject an
  unauthorized request while still never holding a global graph.

Because they are written separately, they can disagree -- and a disagreement is a **directed**
exchange: ``A`` averages ``B``'s model while ``B`` never sees ``A``'s. That is supported, and it
is sometimes the only thing a deployment can do, since a site behind NAT or a one-way firewall
rule can dial out but never be dialed.

It does change the result, though not in the way one might guess. Each node normalizes its
weights over its own incoming set, so the mixing matrix stays row-stochastic and every update
remains a convex combination -- a strongly connected directed graph still reaches consensus.
What it loses is *which* consensus: the limit is weighted by the stationary distribution of the
mixing matrix rather than being the plain average, so a node many others listen to pulls the
result toward itself. Undirected graphs with uniform weights are the case where that bias
vanishes, which is also the case that reproduces centralized FedAvg.

Because of that, asymmetry has to be **declared**: set ``directed: true`` in the block. Lists
that disagree without it are rejected, which catches the realistic error -- someone adds a peer
to ``recv_from`` and forgets ``send_to``, whose only other symptom is a ``PermissionError``
raised at round zero by a *different* node.

For simulation -- one process owning every node, or one rank per node -- writing this out is
pointless ceremony, and at a thousand nodes it is untenable. There the same block names a
graph instead, and the neighbor lists are derived from it::

    neighbors:
      mode: "from_topo"
      topology: "fully_connected"
      topology_kwargs: {}
      num_nodes: 4

Both modes produce the same :class:`Neighbors`, so the agent cannot tell which one its config
used -- which is what keeps the serial run a valid control for the distributed one.

``from_topo`` is a simulation convenience and nothing more. It works only because the node ids
are generated (``Node0``..``Node{num_nodes-1}``), so every node derives the *same* graph and
they agree by construction. A real deployment has no such luxury: sites are named by their
operators, join at different times, and no one holds the roster -- which is why a deployed
node writes ``explicit`` and names only its own neighbors.
"""

from appfl.decentralized.neighbor.base import Neighbors
from appfl.decentralized.neighbor.resolve import (
    build_topology_from_config,
    resolve_all_neighbors_from_topology,
    generated_node_ids,
    resolve_neighbors,
)

__all__ = [
    "Neighbors",
    "resolve_neighbors",
    "resolve_all_neighbors_from_topology",
    "build_topology_from_config",
    "generated_node_ids",
]
