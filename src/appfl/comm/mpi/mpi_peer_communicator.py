from mpi4py import MPI
from typing import Any, Dict, Sequence
from appfl.comm.mpi.serializer import byte_to_model_optimized, model_to_byte_optimized


class MPIPeerCommunicator:
    """
    `MPIPeerCommunicator`
    Exchange payloads with a fixed set of peer ranks, once per round.

    The third role alongside `MPIClientCommunicator` and `MPIServerCommunicator`: in
    centralized federated learning a process is a client or a server, and in a decentralized
    workflow it is a **peer** -- both at once, talking to its graph neighbors rather than to a
    coordinator. There is no server rank, no request/response, and no serve loop.

    :param comm: the MPI communicator, usually `MPI.COMM_WORLD`.
    :param node_id: this process's identity, used to key what peers receive.
    :param node_ids: every participant, in rank order -- `node_ids[r]` is rank `r`'s identity.
    :param send_to: identities that receive this peer's payload.
    :param recv_from: identities whose payloads this peer collects.
    """

    #: Enough to separate consecutive rounds, and no more. The source rank already identifies
    #: the sender, so a tag only has to distinguish rounds -- and a peer can be at most one
    #: round ahead of a neighbor, since publishing `r + 1` requires having received their `r`.
    #: Four leaves margin over that bound while staying far below any implementation's
    #: `MPI_TAG_UB`.
    TAG_CYCLE: int = 4

    #: MPI counts messages with a C `int`, so a single message cannot carry 2 GB or more.
    #: A payload over this is refused with an explanation rather than failing deep inside
    #: a send. It is a property of MPI rather than of any deployment, so it is a constant.
    MAX_MESSAGE_BYTES: int = 2**31 - 1

    def __init__(
        self,
        comm,
        node_id: str,
        node_ids: Sequence[str],
        send_to: Sequence[str],
        recv_from: Sequence[str],
    ) -> None:
        self.comm = comm
        self.node_id = str(node_id)
        self.node_ids = [str(n) for n in node_ids]

        size = comm.Get_size()
        if size != len(self.node_ids):
            raise ValueError(
                f"MPI was started with {size} rank(s) but the federation has "
                f"{len(self.node_ids)} node(s). This communicator gives every node its own "
                f"rank, so the two must match -- a mismatch would silently run a different "
                f"graph than the configuration describes."
            )

        self._rank_of = {node: rank for rank, node in enumerate(self.node_ids)}
        if self.node_id not in self._rank_of:
            raise ValueError(f"node {self.node_id!r} is not among {self.node_ids}")
        if self._rank_of[self.node_id] != comm.Get_rank():
            raise ValueError(
                f"node {self.node_id!r} is rank {self._rank_of[self.node_id]} in the roster "
                f"but this process is rank {comm.Get_rank()}; identities are assigned by "
                f"position, so the launcher must give rank r the id node_ids[r]."
            )

        self._send_ranks = [self._rank(peer) for peer in send_to]
        self._recv_ranks = {str(peer): self._rank(peer) for peer in recv_from}

    def _rank(self, node_id: str) -> int:
        try:
            return self._rank_of[str(node_id)]
        except KeyError:
            raise ValueError(
                f"peer {node_id!r} is not in the roster {self.node_ids}"
            ) from None

    @property
    def rank(self) -> int:
        return self.comm.Get_rank()

    def exchange(self, round_id: int, payload: Any) -> Dict[str, Any]:
        """Send ``payload`` to this peer's out-neighbors and return its in-neighbors'.

        Sends are posted before any receive. That ordering is what keeps the exchange
        deadlock-free: if every rank blocked on a receive first, none would ever send, and a
        payload too large for the eager-send window would hang even a correct-looking program.
        """
        tag = int(round_id) % self.TAG_CYCLE
        data = model_to_byte_optimized(payload)
        if len(data) > self.MAX_MESSAGE_BYTES:
            raise ValueError(
                f"node {self.node_id} tried to send {len(data)} bytes in round {round_id}, "
                f"over the {self.MAX_MESSAGE_BYTES}-byte limit. MPI counts a message with a C "
                f"int, so a single message cannot reach 2 GB; a payload this large needs "
                f"chunking, which this communicator does not yet do."
            )

        # `data` stays referenced until Waitall, so the buffers remain valid in flight.
        requests = [
            self.comm.Isend(data, dest=rank, tag=tag) for rank in self._send_ranks
        ]

        received: Dict[str, Any] = {}
        for peer_id, rank in self._recv_ranks.items():
            status = MPI.Status()
            self.comm.probe(source=rank, tag=tag, status=status)
            buffer = bytearray(status.Get_count(MPI.BYTE))
            self.comm.Recv(buffer, source=rank, tag=tag)
            received[peer_id] = byte_to_model_optimized(bytes(buffer))

        MPI.Request.Waitall(requests)
        return received
