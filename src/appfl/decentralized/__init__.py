"""``appfl.decentralized``

Decentralized federated learning reuses most of APPFL rather than forking it. Anything with a
centralized counterpart lives beside that counterpart, because the adjacency is the point:

===============================  ==================================================
Decentralized                    Centralized counterpart it sits next to
===============================  ==================================================
``appfl.agent.DFLNodeAgent``     ``appfl.agent.ClientAgent`` -- a node *is* a client,
                                 plus the aggregation a server would have done
``appfl.algorithm.aggregator``   ``FedAvgAggregator`` -- the same average, taken over
``.DFLNodeFedAvgAggregator``     a neighborhood instead of the whole federation
``appfl.comm.*``                 the gRPC / MPI communicators already there
===============================  ==================================================

This package holds the rest: the concepts regular FL has no version of. Import them from
their own subpackage rather than from here, so a reader can see which one a dependency is on::

    from appfl.decentralized.neighbor import Neighbors, resolve_neighbors
    from appfl.decentralized.topology import Topology, build_topology

* :mod:`appfl.decentralized.neighbor` -- who a node exchanges models with. This is the whole of
  what a node knows about the graph, and it is deliberately local: the peers it fetches from,
  and the peers it is willing to serve. A deployed site holds no roster and cannot see the
  federation beyond its own edges.
* :mod:`appfl.decentralized.topology` -- the communication graph, for the cases where something
  legitimately owns the whole federation: a simulation launcher, or a node config in
  ``from_topo`` mode. Centralized FL has no graph over its participants at all: clients never
  exchange anything with one another, and no scheduler or aggregator ever asks who a client's
  neighbors are. In DFL the graph *is* the algorithm's behavior.

Algorithms that are themselves decentralized go in ``appfl.decentralized.algorithm``.
"""
