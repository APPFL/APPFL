import time
import grpc
import yaml
import logging
import threading
from concurrent import futures
from typing import Any, Optional
from appfl.comm.grpc.grpc_communicator_pb2 import (
    CustomActionRequest,
    CustomActionResponse,
    GetGlobalModelRespone,
    ServerHeader,
    ServerStatus,
    UpdateGlobalModelRequest,
    UpdateGlobalModelResponse,
)
from appfl.comm.grpc.grpc_communicator_pb2_grpc import (
    GRPCCommunicatorServicer,
    add_GRPCCommunicatorServicer_to_server,
)
from appfl.comm.grpc.payload_store import PayloadStore
from appfl.comm.grpc.utils import (
    MAX_RECEIVE_MESSAGE_BYTES,
    proto_to_databuffer,
    response_chunk_size,
)


class GRPCRelayServicer(GRPCCommunicatorServicer):
    """A switchboard for nodes that cannot be dialled.

    A site behind NAT or a one-way firewall can reach out but never be reached, so it cannot
    serve its own payloads. It publishes them here instead, and the peers that collect from it
    fetch them here. Nothing else changes: a node that can serve still does, and the two kinds
    mix freely in one federation.

    **It holds no graph.** Access control is assembled from what each publisher declares when
    it publishes -- "serve this round to these peers" -- so the relay learns exactly the edges
    it is told about and nothing more. Changing the topology therefore never touches the relay,
    and a relay cannot be asked for a payload nobody authorized it to hand over.
    """

    def __init__(
        self,
        max_message_size: int = 2 * 1024 * 1024,
        wait_timeout: float = 3600.0,
        history: int = 2,
        expected_nodes: Optional[int] = None,
        logger: Optional[Any] = None,
    ):
        self.max_message_size = max_message_size
        self.wait_timeout = wait_timeout
        self.history = history
        self.expected_nodes = expected_nodes
        self.logger = logger if logger is not None else logging.getLogger(__name__)

        self._stores: dict[str, PayloadStore] = {}
        self._allowed: dict[str, set[str]] = {}
        self._seen: set[str] = set()
        self._closed: set[str] = set()
        self._lock = threading.Lock()

    def _denied(self, target: str, requester: str) -> bool:
        """Whether ``target``'s access list excludes ``requester``.

        An absent or empty list means the relay has no opinion to apply: a publisher that
        named no ``send_to`` is readable by anyone, and a target that has not published yet
        has not declared a list at all.
        """
        with self._lock:
            allowed = self._allowed.get(target)
        return bool(allowed) and requester not in allowed

    def _store_for(self, node_id: str) -> PayloadStore:
        with self._lock:
            if node_id not in self._stores:
                self._stores[node_id] = PayloadStore(history=self.history)
            return self._stores[node_id]

    def UpdateGlobalModel(self, request_iterator, context):
        """A node publishes one round's payload. **Not a real global model -- see the peer servicer.**

        `header.client_id` is the publisher, `local_model` the opaque bytes, and `meta_data`
        carries ``round_id`` plus the ``send_to`` list that becomes this publisher's access
        list. Declaring it on every publish is what lets the relay enforce a graph it was never given.
        """
        request = UpdateGlobalModelRequest()
        received = b""
        for chunk in request_iterator:
            received += chunk.data_bytes
        request.ParseFromString(received)

        publisher = request.header.client_id
        meta_data = yaml.safe_load(request.meta_data) if request.meta_data else {}
        round_id = int(meta_data.get("round_id", 0))
        send_to = [str(peer) for peer in (meta_data.get("send_to") or [])]

        store = self._store_for(publisher)
        with self._lock:
            self._allowed[publisher] = set(send_to)
            self._seen.add(publisher)
        store.publish(round_id, request.local_model)
        self.logger.info(
            f"[relay] {publisher} published round {round_id} "
            f"({len(request.local_model)} bytes) for {sorted(send_to) or 'anyone'}"
        )

        response = UpdateGlobalModelResponse(header=ServerHeader(status=ServerStatus.RUN))
        yield from proto_to_databuffer(
            response,
            max_message_size=response_chunk_size(self.max_message_size, meta_data),
        )

    def GetGlobalModel(self, request, context):
        """A node collects a peer's payload. ``meta_data.target`` names whose."""
        requester = request.header.client_id
        meta_data = yaml.safe_load(request.meta_data) if request.meta_data else {}
        round_id = int(meta_data.get("round_id", 0))
        target = str(meta_data.get("target", ""))
        if not target:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "a relay needs meta_data.target naming whose payload is wanted",
            )

        with self._lock:
            self._seen.add(requester)

        # Necessary but not sufficient check first for faster rejection.
        if self._denied(target, requester):
            context.abort(
                grpc.StatusCode.PERMISSION_DENIED,
                f"{target} did not authorize {requester} to collect its payloads",
            )

        store = self._store_for(target)
        try:
            data = store.get(round_id, self.wait_timeout)
        except KeyError as exc:
            context.abort(grpc.StatusCode.OUT_OF_RANGE, str(exc))
        except TimeoutError as exc:
            context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, str(exc))

        # The enforcement access check point
        if self._denied(target, requester):
            context.abort(
                grpc.StatusCode.PERMISSION_DENIED,
                f"{target} did not authorize {requester} to collect its payloads",
            )
        response = GetGlobalModelRespone(
            header=ServerHeader(status=ServerStatus.RUN),
            global_model=data,
            meta_data=yaml.dump({"round_id": round_id, "target": target}),
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

        if request.action != "close_connection":
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                f"the relay does not implement action {request.action!r}",
            )
        meta_data = yaml.safe_load(request.meta_data) if request.meta_data else {}
        node_id = request.header.client_id
        with self._lock:
            self._seen.add(node_id)
            self._closed.add(node_id)
            closed, seen = len(self._closed), len(self._seen)
        self.logger.info(
            f"[relay] {node_id} finished ({closed}/{self.expected_nodes or seen} done)"
        )
        response = CustomActionResponse(header=ServerHeader(status=ServerStatus.DONE))
        yield from proto_to_databuffer(
            response,
            max_message_size=response_chunk_size(self.max_message_size, meta_data),
        )

    def everyone_finished(self) -> bool:
        """Whether the relay has nothing left to do."""
        with self._lock:
            if self.expected_nodes is not None:
                return len(self._closed) >= self.expected_nodes
            return bool(self._seen) and self._closed >= self._seen


def serve_relay(
    servicer: GRPCRelayServicer,
    server_uri: str,
    use_ssl: bool = False,
    server_certificate: Optional[str] = None,
    server_certificate_key: Optional[str] = None,
    max_workers: int = 64,
    poll_seconds: float = 5.0,
    max_receive_message_size: int = MAX_RECEIVE_MESSAGE_BYTES,
) -> None:
    """Run a relay until every node it is waiting on has finished, then stop."""
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ("grpc.max_concurrent_streams", max_workers),
            ("grpc.max_send_message_length", servicer.max_message_size),
            ("grpc.max_receive_message_length", max_receive_message_size),
            ("grpc.keepalive_time_ms", 60000),
            ("grpc.keepalive_timeout_ms", 20000),
            ("grpc.keepalive_permit_without_calls", 1),
        ],
    )
    add_GRPCCommunicatorServicer_to_server(servicer, server)
    if use_ssl:
        from appfl.comm.grpc.utils import load_credential_from_file

        key, certificate = server_certificate_key, server_certificate
        if isinstance(key, str):
            key = load_credential_from_file(key)
        if isinstance(certificate, str):
            certificate = load_credential_from_file(certificate)
        server.add_secure_port(server_uri, grpc.ssl_server_credentials(((key, certificate),)))
    else:
        server.add_insecure_port(server_uri)
    server.start()
    servicer.logger.info(
        f"[relay] listening at {server_uri} with {max_workers} worker(s); "
        + (
            f"stopping once {servicer.expected_nodes} node(s) have finished"
            if servicer.expected_nodes is not None
            else "stopping once every node it has heard from has finished"
        )
    )
    try:
        while not servicer.everyone_finished():
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        servicer.logger.info("[relay] interrupted")
    servicer.logger.info("[relay] all nodes finished; stopping")
    server.stop(0)
