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
    UpdateGlobalModelRequest,
    UpdateGlobalModelResponse,
)
from appfl.comm.grpc.grpc_communicator_pb2_grpc import (
    GRPCCommunicatorServicer,
    GRPCCommunicatorStub,
    add_GRPCCommunicatorServicer_to_server,
)
from appfl.comm.grpc.payload_store import PayloadStore
from appfl.comm.grpc.utils import (
    deserialize_model,
    MAX_RECEIVE_MESSAGE_BYTES,
    proto_to_databuffer,
    response_chunk_size,
    serialize_model,
)


class GRPCPeerServicer(GRPCCommunicatorServicer):
    """Serves this peer's published payloads to the peers allowed to ask for them."""

    def __init__(
        self,
        node_id: str,
        store: PayloadStore,
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
        return not self.send_to or requester_id in self.send_to

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
        target = str(meta_data.get("target", "") or self.node_id)
        if target != self.node_id:
            context.abort(
                grpc.StatusCode.NOT_FOUND,
                f"{self.node_id} serves only its own payloads, not {target}'s",
            )

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
        yield from proto_to_databuffer(
            response,
            max_message_size=response_chunk_size(self.max_message_size, meta_data),
        )

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
            self._closed.add(requester)
            remaining = len(set(self.send_to) - self._closed)
        self.logger.info(
            f"[{self.node_id}] {requester} finished; {remaining} peer(s) still collecting"
        )
        meta_data = yaml.safe_load(request.meta_data) if request.meta_data else {}
        response = CustomActionResponse(header=ServerHeader(status=ServerStatus.DONE))
        yield from proto_to_databuffer(
            response,
            max_message_size=response_chunk_size(self.max_message_size, meta_data),
        )

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
        send_to: Sequence[str],
        recv_from: Mapping[str, str],
        server_uri: Optional[str] = None,
        relay_server_uri: Optional[str] = None,
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
        # Exactly one of `server_uri` or `relay_server_uri`: a node either serves its own payloads or 
        # publishes them to a relay. A site behind NAT can only do the second.
        self.server_uri = server_uri
        self.relay_server_uri = relay_server_uri
        if server_uri and relay_server_uri:
            raise ValueError(
                f"{node_id} sets both server_uri ({server_uri}) and relay_server_uri "
                f"({relay_server_uri}). A node either serves its own payloads or publishes "
                f"them to a relay; doing both would put the same payload in two places with "
                f"no rule for which peers read which."
            )
        if not server_uri and not relay_server_uri and send_to:
            raise ValueError(
                f"{node_id} is expected to serve {sorted(send_to)} but has neither a "
                f"server_uri to listen on nor a relay_server_uri to publish to, so nothing "
                f"could ever collect from it."
            )
        # Ids are normalized to str here, at the boundary where they arrive from YAML --
        # `send_to: [0, 1]` is a perfectly ordinary config. Ids read back off the wire are
        # protobuf `string` fields and are already str, so they are compared as they come.
        self.send_to = [str(peer) for peer in send_to]
        self.recv_from = {str(peer): uri for peer, uri in recv_from.items()}
        # Needed either way: every node dials out, whether or not anything dials it.
        self.use_ssl = use_ssl
        self.root_certificate = root_certificate
        self.max_message_size = max_message_size
        self.connect_timeout = connect_timeout
        self.wait_timeout = wait_timeout
        self.logger = logger if logger is not None else logging.getLogger(__name__)

        if self.server_uri:
            # Serving: a thread pool sized to the peers that collect, a store to serve them
            # from, the servicer that reads it, and the certificate to present.
            self.server_certificate = server_certificate
            self.server_certificate_key = server_certificate_key
            self.max_workers = self._resolve_max_workers(max_workers)
            self.store = PayloadStore()
            self.servicer = GRPCPeerServicer(
                self.node_id,
                self.store,
                self.send_to,
                max_message_size=max_message_size,
                wait_timeout=wait_timeout,
                logger=self.logger,
            )
            self._server = None
        # Keyed by endpoint rather than by peer: several peers reached through one relay share
        # a single channel, and the relay this node publishes to is usually one of them.
        self._channels: Dict[str, Any] = {}
        self._stubs: Dict[str, GRPCCommunicatorStub] = {}

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
        if self.server_uri:
            self._server = grpc.server(
                futures.ThreadPoolExecutor(max_workers=self.max_workers),
                options=[
                    ("grpc.max_concurrent_streams", self.max_workers),
                    ("grpc.max_send_message_length", self.max_message_size),
                    ("grpc.max_receive_message_length", MAX_RECEIVE_MESSAGE_BYTES),
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
                f"[{self.node_id}] serving at {self.server_uri} with "
                f"{self.max_workers} worker(s); "
                f"{len(self.send_to)} peer(s) may collect from it"
            )
        else:
            self.logger.info(
                f"[{self.node_id}] not serving; publishing to the relay at "
                f"{self.relay_server_uri}. Nothing dials this node, which is the point -- a "
                f"site that cannot accept inbound connections still participates fully."
            )

        endpoints = dict.fromkeys(
            [uri for uri in self.recv_from.values() if uri]
            + ([self.relay_server_uri] if self.relay_server_uri else [])
        )
        for uri in endpoints:
            channel = create_grpc_channel(
                uri,
                use_ssl=self.use_ssl,
                root_certificate=self.root_certificate,
                max_message_size=self.max_message_size,
            )
            grpc.channel_ready_future(channel).result(timeout=self.connect_timeout)
            self._channels[uri] = channel
            self._stubs[uri] = GRPCCommunicatorStub(channel)
            # One endpoint can serve both roles at once, and usually does: two relay-using
            # nodes collect each other's payloads through the same relay they publish to.
            roles = []
            if uri == self.relay_server_uri:
                roles.append("publishing through it")
            reached = sorted(p for p, u in self.recv_from.items() if u == uri)
            if reached:
                roles.append(f"collecting {reached} from it")
            self.logger.info(
                f"[{self.node_id}] connected to {uri}: {', '.join(roles)}"
            )
        self.logger.info(
            f"[{self.node_id}] ready: collecting from {sorted(self.recv_from)}"
        )

    def shutdown(self) -> None:
        """Tell the peers this node collects from that it is done, then stop serving.

        It keeps serving until every peer it serves has said the same, so a neighbor still
        waiting on this node's last round is not cut off mid-collection.
        """
        for uri, stub in self._stubs.items():
            try:
                request = CustomActionRequest(
                    header=ClientHeader(client_id=self.node_id),
                    action="close_connection",
                    meta_data=yaml.dump({"max_message_size": self.max_message_size}),
                )
                received = b""
                for chunk in stub.InvokeCustomAction(
                    proto_to_databuffer(
                        request, max_message_size=self.max_message_size
                    ),
                    timeout=60,
                ):
                    received += chunk.data_bytes
                self.logger.info(f"[{self.node_id}] told {uri} it is finished")
            except grpc.RpcError as error:
                # A peer that has already stopped is not an error worth failing a run over.
                self.logger.warning(
                    f"[{self.node_id}] could not reach {uri} while closing: "
                    f"{error.code().name}"
                )

        # Only a node that serves has collectors to wait for; one publishing through a relay
        # has already handed everything over and owes nobody anything.
        if self.server_uri and self.send_to:
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
        if self.server_uri:
            self._server.stop(0)
        self.logger.info(f"[{self.node_id}] stopped")

    def __enter__(self) -> "GRPCPeerCommunicator":
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.shutdown()

    # -- the exchange ------------------------------------------------------------------

    def exchange(self, round_id: int, payload: Any) -> Dict[str, Any]:
        """Publish ``payload`` for this round and return the neighbors' payloads.

        Publishing goes wherever this node is reachable -- its own store if it serves, the
        relay if it does not. Collecting is identical either way: the request names the peer
        whose payload is wanted, and a self-hosting peer ignores that field while a relay uses
        it to look up. So the fetching side never has to know which kind of endpoint it is
        talking to, which is what lets the two mix in one federation.
        """
        data = serialize_model(payload)
        if self.relay_server_uri:
            self._publish_to_relay(int(round_id), data)
        else:
            self.store.publish(int(round_id), data)

        received: Dict[str, Any] = {}
        for peer_id, uri in self.recv_from.items():
            request = GetGlobalModelRequest(
                header=ClientHeader(client_id=self.node_id),
                # The chunk size has to fit the *receiver*, so this says what it can take.
                # A relay serving several nodes cannot assume they all agree, and a chunk
                # above the requester's channel limit is refused after it is already sent.
                meta_data=yaml.dump(
                    {
                        "round_id": int(round_id),
                        "target": peer_id,
                        "max_message_size": self.max_message_size,
                    }
                ),
            )
            chunks = b""
            for chunk in self._stubs[uri].GetGlobalModel(
                request, timeout=self.wait_timeout
            ):
                chunks += chunk.data_bytes
            response = GetGlobalModelRespone()
            response.ParseFromString(chunks)
            if response.header.status == ServerStatus.ERROR:
                raise RuntimeError(f"{peer_id} returned an error for round {round_id}")
            received[peer_id] = deserialize_model(response.global_model)
        return received

    def _publish_to_relay(self, round_id: int, data: bytes) -> None:
        """Hand this round's payload to the relay, with the access list that applies to it.

        `send_to` travels with every publish so the relay can enforce a graph it was never
        given -- it learns only the edges its publishers tell it about.
        """
        request = UpdateGlobalModelRequest(
            header=ClientHeader(client_id=self.node_id),
            local_model=data,
            meta_data=yaml.dump(
                {
                    "round_id": round_id,
                    "send_to": list(self.send_to),
                    "max_message_size": self.max_message_size,
                }
            ),
        )
        received = b""
        for chunk in self._stubs[self.relay_server_uri].UpdateGlobalModel(
            proto_to_databuffer(request, max_message_size=self.max_message_size),
            timeout=self.wait_timeout,
        ):
            received += chunk.data_bytes
        response = UpdateGlobalModelResponse()
        response.ParseFromString(received)
        if response.header.status == ServerStatus.ERROR:
            raise RuntimeError(f"the relay rejected round {round_id} from {self.node_id}")
