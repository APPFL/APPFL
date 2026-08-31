"""One agent per MPI rank, peer-to-peer. The HPC scaling backend."""

from __future__ import annotations

from typing import List, Sequence

from appfl.decentralized.exchange.base import TokenExchange
from appfl.decentralized.exchange.codec import pack_tokens, unpack_tokens
from appfl.decentralized.protocol import TokenProtocol
from appfl.decentralized.topology import Topology


class MPIExchange(TokenExchange):
    """One agent per MPI rank, peer-to-peer. The HPC scaling path.

    Ranks map to ``topology.agent_ids`` by position, so rank ``r`` is
    ``topology.agent_ids[r]`` and the communicator size must equal the agent count.

    Sends are non-blocking (``isend``) and receives are posted for exactly the set of
    in-neighbors the topology names, so a round costs ``2 * |E|`` messages and no collectives.
    That is the point: no rank is a coordinator, and cost scales with graph degree rather
    than agent count -- which is what makes a sparse topology cheaper than a dense one in
    wall-clock as well as in bits, and what the coordination-scaling study needs to measure.

    Requires ``mpi4py`` (APPFL's ``[mpi]`` extra).
    """

    owns_single_agent = True

    def __init__(self, topology: Topology, comm=None, budget=None, meter=None, token_cls=None):
        super().__init__(topology, budget, meter, token_cls)
        from mpi4py import MPI

        self.MPI = MPI
        self.comm = comm if comm is not None else MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        size = self.comm.Get_size()
        if size != len(topology.agent_ids):
            raise ValueError(
                f"MPI communicator size {size} != agent count "
                f"{len(topology.agent_ids)}. Launch with -n {len(topology.agent_ids)}."
            )
        self.agent_id = topology.agent_ids[self.rank]
        self._rank_of = {a: i for i, a in enumerate(topology.agent_ids)}
        # In-neighbors: who will send to me. Distinct from out-neighbors on a directed
        # topology, and computing it once avoids probing for messages that never come.
        self._in_neighbors = [
            a for a in topology.agent_ids if self.agent_id in topology.neighbors(a)
        ]
        self._pending: List = []
        self._received: List[TokenProtocol] = []

    TAG = 7717  # arbitrary, distinct from APPFL's task tags

    def publish(self, agent_id: str, tokens: Sequence[TokenProtocol]) -> None:
        if agent_id != self.agent_id:
            raise ValueError(
                f"rank {self.rank} owns agent {self.agent_id!r}, not {agent_id!r}"
            )
        admitted = self._admit(tokens)
        payload = pack_tokens(admitted)
        for peer in self.topology.neighbors(agent_id):
            # Non-blocking so a slow neighbor does not serialize the whole round; the
            # requests are drained in barrier().
            self._pending.append(
                self.comm.isend(payload, dest=self._rank_of[peer], tag=self.TAG)
            )
            self.meter.record_sent(admitted)

    def collect(self, agent_id: str) -> List[TokenProtocol]:
        received, self._received = self._received, []
        return received

    def barrier(self, round_idx: int = 0) -> None:
        """Complete this round's sends and take delivery of every in-neighbor's tokens.

        ``round_idx`` is unused here -- MPI's own message ordering and ``Barrier`` already
        make the round boundary unambiguous.

        Receives are blocking and counted, which is what makes the round well-defined: an
        agent has heard from all of its in-neighbors or the round has not ended. Bounded
        staleness -- letting an agent proceed on whatever arrived -- is the asynchronous
        variant, and is open question 3 in the README.
        """
        for _ in self._in_neighbors:
            payload = self.comm.recv(source=self.MPI.ANY_SOURCE, tag=self.TAG)
            self._received.extend(unpack_tokens(payload, self.token_cls))
        self.MPI.Request.waitall(self._pending)
        self._pending = []
        self.comm.Barrier()

    def gather_results(self, local_result):
        """Collect any picklable per-agent result on rank 0. For the experiment driver."""
        return self.comm.gather(local_result, root=0)
