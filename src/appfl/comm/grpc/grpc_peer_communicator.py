import time
import grpc
import yaml
import logging
import threading
from concurrent import futures
from typing import Any, Dict, Mapping, Optional, Sequence
from appfl.comm.grpc.channel import create_grpc_channel
from appfl.comm.grpc.grpc_communicator_pb2 import (
    ClientHeader,
    CustomActionRequest,
    CustomActionResponse,
    GetGlobalModelRequest,
    GetGlobalModelRespone,
    ServerHeader,
    ServerStatus,
)
from appfl.comm.grpc.grpc_communicator_pb2_grpc import (
    GRPCCommunicatorServicer,
    GRPCCommunicatorStub,
    add_GRPCCommunicatorServicer_to_server,
)
from appfl.comm.grpc.utils import (
    deserialize_model,
    proto_to_databuffer,
    serialize_model,
)


class _PayloadStore:
    """What this peer has published, by round, as bytes ready to send."""

    def __init__(self, history: int = 2):
        self.history = history
        self._rounds: Dict[int, bytes] = {}
        self._latest = -1
        self._condition = threading.Condition()

    def publish(self, round_id: int, data: bytes) -> None:
        with self._condition:
            self._rounds[round_id] = data
            self._latest = max(self._latest, round_id)
            for stale in sorted(self._rounds)[: -self.history]:
                del self._rounds[stale]
            self._condition.notify_all()

    def get(self, round_id: int, timeout: float) -> bytes:
        """Block until ``round_id`` is published, then return it."""
        deadline = time.monotonic() + timeout
        with self._condition:
            while round_id not in self._rounds:
                if round_id < self._latest:
                    raise KeyError(
                        f"round {round_id} was asked for but this peer has already moved past "
                        f"it (now at {self._latest}, keeping {self.history}). The requester "
                        f"has fallen further behind than the synchronous protocol allows."
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._condition.wait(remaining):
                    raise TimeoutError(
                        f"round {round_id} was not published within {timeout:.0f}s"
                    )
            return self._rounds[round_id]


class GRPCPeerServicer(GRPCCommunicatorServicer):
    """Serves this peer's published payloads to the peers allowed to ask for them."""

    def __init__(
        self,
        node_id: str,
        store: _PayloadStore,
        send_to: Sequence[str],
        max_message_size: int,
        wait_timeout: float,
        logger: Optional[Any] = None,
    ):
        self.node_id = node_id
        self.store = store
        self.send_to = [str(peer) for peer in send_to]
        self.max_message_size = max_message_size
        self.wait_timeout = wait_timeout
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self._closed = set()
        self._closed_lock = threading.Lock()

    def _may_serve(self, requester_id: str) -> bool:
        """An empty `send_to` serves anyone; otherwise only the peers this node declared.

        The policy itself lives on :class:`~appfl.decentralized.neighbor.Neighbors`, which is
        where a caller should ask about it; this is only how the servicer applies it to a
        request that has arrived.
        """
        return not self.send_to or str(requester_id) in self.send_to

    def GetGlobalModel(self, request, context):
        """Return this peer's payload for a requested round. **Nothing global is involved.**

        The name is inherited from the proto, where it means "the client fetches the server's
        global model" -- the centralized operation this file has no equivalent of. Here there
        is no global model and no server: one peer is asking another for the payload it
        published in a particular round, and a peer two hops away may be asking for a
        different round at the same moment.

        The fields are reused the same way, so the mapping is worth stating outright:

        * ``request.header.client_id`` -- the *requesting* peer's id, checked against
          this node's ``send_to``. Not a client in any sense the proto meant.
        * ``request.meta_data`` -- YAML carrying ``round_id``, which round is wanted.
        * ``response.global_model`` -- this peer's serialized payload for that round.
        * ``response.meta_data`` -- the round actually served, echoed back.
        """
        requester = request.header.client_id
        meta_data = yaml.safe_load(request.meta_data) if request.meta_data else {}
        round_id = int(meta_data.get("round_id", 0))

        if not self._may_serve(requester):
            # Enforced at the transport because that is where the request arrives from the
            # network; the declaration itself lives in this node's own configuration.
            context.abort(
                grpc.StatusCode.PERMISSION_DENIED,
                f"{self.node_id} does not serve {requester}: not in its send_to list",
            )
        try:
            data = self.store.get(round_id, self.wait_timeout)
        except KeyError as exc:
            context.abort(grpc.StatusCode.OUT_OF_RANGE, str(exc))
        except TimeoutError as exc:
            context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, str(exc))

        response = GetGlobalModelRespone(
            header=ServerHeader(status=ServerStatus.RUN),
            global_model=data,
            meta_data=yaml.dump({"round_id": round_id}),
        )
        yield from proto_to_databuffer(response, max_message_size=self.max_message_size)

    def InvokeCustomAction(self, request_iterator, context):
        request = CustomActionRequest()
        received = b""
        for chunk in request_iterator:
            received += chunk.data_bytes
        request.ParseFromString(received)

        requester = request.header.client_id
        if request.action != "close_connection":
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                f"{self.node_id} does not implement action {request.action!r}",
            )
        with self._closed_lock:
            self._closed.add(str(requester))
            remaining = len(set(self.send_to) - self._closed)
        self.logger.info(
            f"[{self.node_id}] {requester} finished; {remaining} peer(s) still collecting"
        )
        response = CustomActionResponse(header=ServerHeader(status=ServerStatus.DONE))
        yield from proto_to_databuffer(response, max_message_size=self.max_message_size)

    def everyone_finished(self) -> bool:
        """Whether every peer this node serves has said it is done collecting."""
        with self._closed_lock:
            return not set(self.send_to) - self._closed


class GRPCPeerCommunicator:
    """
    `GRPCPeerCommunicator`
    Exchange payloads with graph neighbors over gRPC, once per round.

    :meth:`exchange` takes an opaque payload and returns the neighbors' payloads.
    Each peer both serves -- its own gRPC server, answering the neighbors that collect from it
    -- and dials out to the neighbors it collects from. There is no coordinator.

    :param node_id: this peer's identity, sent with every request so peers can key what they
        receive and enforce who they serve.
    :param server_uri: where this peer listens.
    :param send_to: identities permitted to collect this peer's payloads. Empty serves anyone.
    :param recv_from: ``{node_id: server_uri}`` for the peers this one collects from.
    :param max_workers: servicer threads. Derived from the number of peers served when not
        given, since each blocked request holds a thread for as long as the round takes.
    """

    #: Spare threads beyond one per served peer: close-connection calls arrive on the same
    #: pool, and gRPC itself needs room to answer while requests sit blocked on a round.
    WORKER_HEADROOM: int = 4
    #: Never fewer than this, so a peer with one neighbor still has room to be talked to.
    MIN_WORKERS: int = 8

    def __init__(
        self,
        node_id: str,
        server_uri: str,
        send_to: Sequence[str],
        recv_from: Mapping[str, str],
        use_ssl: bool = False,
        server_certificate: Optional[str] = None,
        server_certificate_key: Optional[str] = None,
        root_certificate: Optional[str] = None,
        max_message_size: int = 2 * 1024 * 1024,
        max_workers: Optional[int] = None,
        connect_timeout: float = 3600.0,
        wait_timeout: float = 3600.0,
        logger: Optional[Any] = None,
    ) -> None:
        self.node_id = str(node_id)
        self.server_uri = server_uri
        self.send_to = [str(peer) for peer in send_to]
        self.recv_from = {str(peer): uri for peer, uri in recv_from.items()}
        self.use_ssl = use_ssl
        self.server_certificate = server_certificate
        self.server_certificate_key = server_certificate_key
        self.root_certificate = root_certificate
        self.max_message_size = max_message_size
        self.connect_timeout = connect_timeout
        self.wait_timeout = wait_timeout
        self.logger = logger if logger is not None else logging.getLogger(__name__)

        self.max_workers = self._resolve_max_workers(max_workers)
        self.store = _PayloadStore()
        self.servicer = GRPCPeerServicer(
            self.node_id,
            self.store,
            self.send_to,
            max_message_size=max_message_size,
            wait_timeout=wait_timeout,
            logger=self.logger,
        )
        self._server = None
        self._stubs: Dict[str, GRPCCommunicatorStub] = {}
        self._channels: Dict[str, Any] = {}

    def _resolve_max_workers(self, max_workers: Optional[int]) -> int:
        """One thread per peer served, plus headroom, unless the caller insists otherwise.

        Sized automatically because getting it wrong starves rather than fails: a peer with
        more neighbors than threads leaves some of them blocked in the queue, and the
        federation stalls with no error anywhere.
        """
        needed = len(self.send_to) + self.WORKER_HEADROOM
        if max_workers is None:
            return max(self.MIN_WORKERS, needed)
        if max_workers <= len(self.send_to):
            self.logger.warning(
                f"[{self.node_id}] max_workers={max_workers} is not more than the "
                f"{len(self.send_to)} peer(s) this node serves. Each collecting peer holds a "
                f"thread until the round it waits for is published, so the federation can "
                f"stall with no error reported. At least {needed} is recommended."
            )
        return max_workers

    # -- lifecycle ---------------------------------------------------------------------

    def start(self) -> None:
        """Start serving, then dial every peer this node collects from."""
        self._server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self.max_workers),
            options=[
                ("grpc.max_concurrent_streams", self.max_workers),
                ("grpc.max_send_message_length", self.max_message_size),
                ("grpc.max_receive_message_length", self.max_message_size),
                ("grpc.keepalive_time_ms", 60000),
                ("grpc.keepalive_timeout_ms", 20000),
                ("grpc.keepalive_permit_without_calls", 1),
            ],
        )
        add_GRPCCommunicatorServicer_to_server(self.servicer, self._server)
        if self.use_ssl:
            from appfl.comm.grpc.utils import load_credential_from_file

            key = self.server_certificate_key
            certificate = self.server_certificate
            if isinstance(key, str):
                key = load_credential_from_file(key)
            if isinstance(certificate, str):
                certificate = load_credential_from_file(certificate)
            self._server.add_secure_port(
                self.server_uri, grpc.ssl_server_credentials(((key, certificate),))
            )
        else:
            self._server.add_insecure_port(self.server_uri)
        self._server.start()
        self.logger.info(
            f"[{self.node_id}] serving at {self.server_uri} with {self.max_workers} worker(s); "
            f"{len(self.send_to)} peer(s) may collect from it"
        )

        for peer_id, uri in self.recv_from.items():
            channel = create_grpc_channel(
                uri,
                use_ssl=self.use_ssl,
                root_certificate=self.root_certificate,
                max_message_size=self.max_message_size,
            )
            grpc.channel_ready_future(channel).result(timeout=self.connect_timeout)
            self._channels[peer_id] = channel
            self._stubs[peer_id] = GRPCCommunicatorStub(channel)
            self.logger.info(f"[{self.node_id}] connected to {peer_id} at {uri}")
        self.logger.info(
            f"[{self.node_id}] ready: collecting from {sorted(self.recv_from)}"
        )

    def shutdown(self) -> None:
        """Tell the peers this node collects from that it is done, then stop serving.

        It keeps serving until every peer it serves has said the same, so a neighbor still
        waiting on this node's last round is not cut off mid-collection.
        """
        for peer_id, stub in self._stubs.items():
            try:
                request = CustomActionRequest(
                    header=ClientHeader(client_id=self.node_id),
                    action="close_connection",
                    meta_data=yaml.dump({}),
                )
                received = b""
                for chunk in stub.InvokeCustomAction(
                    proto_to_databuffer(
                        request, max_message_size=self.max_message_size
                    ),
                    timeout=60,
                ):
                    received += chunk.data_bytes
                self.logger.info(f"[{self.node_id}] told {peer_id} it is finished")
            except grpc.RpcError as error:
                # A peer that has already stopped is not an error worth failing a run over.
                self.logger.warning(
                    f"[{self.node_id}] could not reach {peer_id} while closing: "
                    f"{error.code().name}"
                )

        if self.send_to:
            self.logger.info(
                f"[{self.node_id}] waiting for {len(self.send_to)} peer(s) to finish "
                f"collecting before stopping"
            )
            deadline = time.monotonic() + self.wait_timeout
            while not self.servicer.everyone_finished():
                if time.monotonic() > deadline:
                    self.logger.warning(
                        f"[{self.node_id}] stopping with peers still collecting; they may see "
                        f"a failed request"
                    )
                    break
                time.sleep(0.1)

        for channel in self._channels.values():
            channel.close()
        if self._server is not None:
            self._server.stop(0)
        self.logger.info(f"[{self.node_id}] stopped")

    def __enter__(self) -> "GRPCPeerCommunicator":
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.shutdown()

    # -- the exchange ------------------------------------------------------------------

    def exchange(self, round_id: int, payload: Any) -> Dict[str, Any]:
        """Publish ``payload`` for this round and return the neighbors' payloads."""
        self.store.publish(int(round_id), serialize_model(payload))

        received: Dict[str, Any] = {}
        for peer_id, stub in self._stubs.items():
            request = GetGlobalModelRequest(
                header=ClientHeader(client_id=self.node_id),
                meta_data=yaml.dump({"round_id": int(round_id)}),
            )
            data = b""
            for chunk in stub.GetGlobalModel(request, timeout=self.wait_timeout):
                data += chunk.data_bytes
            response = GetGlobalModelRespone()
            response.ParseFromString(data)
            if response.header.status == ServerStatus.ERROR:
                raise RuntimeError(f"{peer_id} returned an error for round {round_id}")
            received[peer_id] = deserialize_model(response.global_model)
        return received
