"""Token transport -- one interface, three backends, identical agent code under each.

This is what APPFL contributes that the ADKO reference does not have. The reference runs its
agents as objects in one Python process; the same algorithm needs to run as ``mpirun -n N`` on
an HPC system and across genuinely separate institutions whose data cannot move.

:class:`TokenExchange` is the seam. An agent never learns which backend is under it::

    exchange.publish(agent_id, tokens)   # send to my neighbours, subject to the bit budget
    exchange.collect(agent_id)           # tokens my neighbours sent me
    exchange.barrier(round_idx)          # deliver this round's traffic, then synchronize

Every backend defers delivery to ``barrier``. That uniformity is load-bearing: it is what lets
one runner body be correct under all three, so a disagreement between backends means the
transport changed something and not the loop.

What "decentralized" means differs across the three, and the difference matters when
describing this work:

* :class:`InProcessExchange` and :class:`MPIExchange` are decentralized in both senses --
  no central authority in the algorithm, and genuinely peer-to-peer transport.
* :class:`RelayExchange` keeps the algorithm decentralized (no shared model, no pooled
  data, each agent sees only its neighbours' tokens) while routing bytes through a coordinator
  that never inspects them. A deliberate trade: DOE sites generally cannot accept inbound
  connections, so a full gRPC mesh would need N x N firewall exceptions and N certificates.
"""

from appfl.decentralized.exchange.base import TokenExchange
from appfl.decentralized.exchange.codec import (
    default_token_cls,
    pack_tokens,
    unpack_tokens,
)
from appfl.decentralized.exchange.in_process import InProcessExchange
from appfl.decentralized.exchange.mpi import MPIExchange
from appfl.decentralized.exchange.relay import RelayExchange, RelayServer

__all__ = [
    "TokenExchange",
    "InProcessExchange",
    "MPIExchange",
    "RelayExchange",
    "RelayServer",
    "pack_tokens",
    "unpack_tokens",
    "default_token_cls",
]
