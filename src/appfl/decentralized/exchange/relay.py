"""Cross-site transport over APPFL's gRPC channel, and the relay that routes for it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Sequence

from appfl.decentralized.exchange.base import TokenExchange
from appfl.decentralized.exchange.codec import pack_tokens, unpack_tokens
from appfl.decentralized.protocol import TokenProtocol
from appfl.decentralized.topology import Topology

if TYPE_CHECKING:
    from appfl.comm.grpc import GRPCClientCommunicator


class RelayExchange(TokenExchange):
    """Cross-site transport over APPFL's gRPC channel. The real-deployment path.

    Each site runs one agent and dials out to a coordinator running
    :class:`RelayServer`. The relay applies the topology and hands each site only its
    neighbors' tokens. It never sees data, models, or raw design points -- only the same
    tokens the neighbors would have received directly -- so the privacy properties are
    exactly those of the peer-to-peer backends.

    Carried on ``InvokeCustomAction``, which is already in APPFL's proto with SSL and
    authentication wired, so nothing here requires regenerating protobufs or opening a second
    port.

    :param agent_id: the single agent this site owns.
    :param communicator: a connected :class:`~appfl.comm.grpc.GRPCClientCommunicator`. Only
        its ``invoke_custom_action`` is used, so a stub with that method works in tests.
    """

    owns_single_agent = True

    def __init__(
        self,
        topology: Topology,
        agent_id: str,
        communicator: GRPCClientCommunicator,
        budget=None,
        meter=None,
        token_cls=None,
    ):
        super().__init__(topology, budget, meter, token_cls)
        self.agent_id = agent_id
        self.communicator = communicator

    def publish(self, agent_id: str, tokens: Sequence[TokenProtocol]) -> None:
        if agent_id != self.agent_id:
            raise ValueError(f"this site owns agent {self.agent_id!r}, not {agent_id!r}")
        admitted = self._admit(tokens)
        if not admitted:
            return
        self.communicator.invoke_custom_action(
            action="token_publish",
            agent_id=agent_id,
            tokens=pack_tokens(admitted),
        )
        for _ in self.topology.neighbors(agent_id):
            self.meter.record_sent(admitted)

    def collect(self, agent_id: str) -> List[TokenProtocol]:
        response = self.communicator.invoke_custom_action(
            action="token_collect", agent_id=agent_id
        )
        return unpack_tokens((response or {}).get("tokens", []), self.token_cls)

    def barrier(
        self,
        round_idx: int = 0,
        poll_seconds: float = 1.0,
        timeout_seconds: float = 3600.0,
    ) -> None:
        """Wait until every site has reached ``round_idx``.

        Sites differ in evaluation cost -- a beamline measurement and a table lookup are not
        the same wall-clock -- so without this the fast sites race ahead and read stale token
        memory. Keeping rounds aligned is what makes a multi-site run comparable to the MPI
        run that models it.

        Polls rather than holding the RPC open: a blocking server-side barrier ties up one
        gRPC thread per site for the duration of the slowest evaluation, and trips the
        request timeout on any run where a site is slow -- which is every real run.
        """
        import time

        deadline = time.time() + timeout_seconds
        while True:
            response = self.communicator.invoke_custom_action(
                action="token_barrier",
                agent_id=self.agent_id,
                round_idx=round_idx,
            ) or {}
            if response.get("complete"):
                return
            if time.time() > deadline:
                raise TimeoutError(
                    f"agent {self.agent_id!r} waited {timeout_seconds}s at the barrier for "
                    f"round {round_idx}; {response.get('arrived', '?')} of "
                    f"{response.get('expected', '?')} sites have arrived, so a peer is "
                    f"unreachable or stalled"
                )
            time.sleep(poll_seconds)


class RelayServer:
    """Server-side counterpart to :class:`RelayExchange`. Routes; never interprets.

    Wire it into an APPFL gRPC server by dispatching the three ``token_*`` custom actions to
    :meth:`handle`. It holds one queue per agent and a round barrier, and that is all -- no
    aggregation, no global model, no view of anyone's data. The federation is decentralized;
    this process is a switchboard.
    """

    def __init__(self, topology: Topology):
        import threading

        self.topology = topology
        self._queues: Dict[str, List[str]] = {a: [] for a in topology.agent_ids}
        # Round index each agent is currently waiting on, and the highest round every agent
        # has reached. Kept as counters rather than a set so that an agent polling after the
        # barrier already cleared still sees its round as complete.
        self._waiting: Dict[str, int] = {}
        self._completed_round = -1
        self._lock = threading.Lock()  # gRPC dispatches handlers on a thread pool

    def handle(self, action: str, agent_id: str, **kwargs) -> Dict:
        """Dispatch one ``token_*`` custom action. Returns the metadata dict to send back."""
        if agent_id not in self._queues:
            raise ValueError(f"unknown agent {agent_id!r} for this federation")
        with self._lock:
            if action == "token_publish":
                return self._publish(agent_id, kwargs.get("tokens", []))
            if action == "token_collect":
                return self._collect(agent_id)
            if action == "token_barrier":
                # The round index must be forwarded, not defaulted. Dropping it made every
                # barrier look like round 0, so once round 0 completed every later barrier
                # returned complete immediately and the federation ran with no
                # synchronization at all -- a silent failure, since each site still finished.
                return self._barrier(agent_id, int(kwargs.get("round_idx", 0)))
        raise ValueError(f"unknown token action {action!r}")

    def _publish(self, agent_id: str, packed: Sequence[str]) -> Dict:
        for peer in self.topology.neighbors(agent_id):
            self._queues[peer].extend(packed)
        return {"accepted": len(packed)}

    def _collect(self, agent_id: str) -> Dict:
        tokens, self._queues[agent_id] = self._queues[agent_id], []
        return {"tokens": tokens}

    def _barrier(self, agent_id: str, round_idx: int = 0) -> Dict:
        """Register this agent at ``round_idx``; report whether that round has completed.

        The caller supplies the round rather than the server inferring it. That is the whole
        design: a server trying to guess which round a poll refers to cannot distinguish the
        first call of a new barrier from a repeat poll of the current one, and any predicate
        it uses to tell them apart also decides completion -- so the last arriver passes and
        everyone else gets silently advanced into the next round and blocks forever.

        With the round supplied, this is idempotent and order-independent: an agent polling
        for round ``r`` sees ``complete`` exactly once every agent has registered at ``r`` or
        beyond, no matter who asked first or how often.

        Non-blocking by design: holding a gRPC handler open until every site arrives ties up
        one server thread per site and trips the request timeout whenever a site is slow --
        which is every real run. Clients poll :meth:`RelayExchange.barrier` instead.
        """
        self._waiting[agent_id] = max(self._waiting.get(agent_id, -1), round_idx)
        arrived = sum(1 for v in self._waiting.values() if v >= round_idx)
        expected = len(self.topology.agent_ids)
        complete = arrived == expected
        if complete:
            self._completed_round = max(self._completed_round, round_idx)
        return {
            "round": round_idx,
            "complete": complete,
            "arrived": arrived,
            "expected": expected,
        }
