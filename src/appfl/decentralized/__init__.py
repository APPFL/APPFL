"""``appfl.decentralized`` -- decentralized coordination over APPFL's communication layer.

**This package is the generic layer and exports only generic things.** It moves opaque
payloads between agents according to a graph, drives the rounds, caps outgoing traffic, and
records what crossed the wire. It does not know what a payload means.

Algorithms live in subpackages and are imported explicitly::

    from appfl.decentralized import InProcessExchange, run_federation   # infrastructure
    from appfl.decentralized.algorithm.adko import ADKOAgent, KnowledgeToken      # one algorithm

That separation is deliberate rather than tidy-minded: re-exporting ADKO here would let you
depend on an algorithm without noticing you had, and the whole point of the split is that a
second decentralized method should be able to reuse everything below without touching it.

Currently implemented: :mod:`appfl.decentralized.algorithm.adko` -- Agentic Decentralized Knowledge
Optimization (Rillo et al., arXiv:2605.07863), decentralized Bayesian optimization in which
agents hold private surrogates and exchange one compact token per neighbour per round.

The same agent code runs three ways unchanged: in one process, under ``mpirun``, or across
real institutions over gRPC. See ``README.md``.
"""

from appfl.decentralized.protocol import AgentProtocol, TokenProtocol
from appfl.decentralized.topology import (
    FullyConnected,
    RandomGeometric,
    Ring,
    Star,
    Topology,
    build_topology,
)
from appfl.decentralized.budget import CommBudget
from appfl.decentralized.metrics import Meter, metered_evaluator
from appfl.decentralized.exchange import (
    InProcessExchange,
    MPIExchange,
    RelayExchange,
    TokenExchange,
    RelayServer,
    pack_tokens,
    unpack_tokens,
)
from appfl.decentralized.runner import run_federation, run_local_agent

__all__ = [
    # contracts
    "TokenProtocol",
    "AgentProtocol",
    # the graph
    "Topology",
    "FullyConnected",
    "Star",
    "Ring",
    "RandomGeometric",
    "build_topology",
    # transport
    "TokenExchange",
    "InProcessExchange",
    "MPIExchange",
    "RelayExchange",
    "RelayServer",
    "pack_tokens",
    "unpack_tokens",
    # the round loop
    "run_federation",
    "run_local_agent",
    # enforcement and observation
    "CommBudget",
    "Meter",
    "metered_evaluator",
]
