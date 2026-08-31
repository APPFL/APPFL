"""The gRPC server half of the relay protocol.

**Not imported by** ``appfl.decentralized.exchange``, on purpose. Pulling this in requires
``appfl.comm.grpc``, which loads grpc, torch, proxystore and boto3 -- roughly 7,500 modules
and several seconds. A process running only the MPI backend should not pay that, so the cost
lands on the one process that actually serves a relay::

    from appfl.decentralized.exchange.grpc_servicer import RelayServicer
"""

from __future__ import annotations

from typing import Any, Optional

import yaml

from appfl.comm.grpc import proto_to_databuffer
from appfl.comm.grpc.grpc_communicator_pb2 import (
    CustomActionRequest,
    CustomActionResponse,
    ServerHeader,
    ServerStatus,
)
from appfl.comm.grpc.grpc_communicator_pb2_grpc import GRPCCommunicatorServicer

from appfl.decentralized.exchange.relay import RelayServer


class RelayServicer(GRPCCommunicatorServicer):
    """Exposes a :class:`RelayServer` over APPFL's existing gRPC service.

    Implements only ``InvokeCustomAction``. Riding APPFL's proto means SSL, authentication and
    message chunking are inherited rather than reimplemented, and no protobuf regeneration is
    needed to add a decentralized federation to an existing deployment.

    Note what this does *not* construct: a ``ServerAgent``. There is no model, no aggregator,
    no scheduler. That absence is the clearest statement of how little a decentralized run
    needs from a server -- it is a switchboard, not a coordinator.

    :param relay: the routing table and round barrier this servicer exposes.
    :param on_shutdown: called by ``serve`` on exit with the action count. Left to the caller
        so the library does not print; an example can pass ``print``.
    """

    def __init__(
        self,
        relay: RelayServer,
        max_message_size: int = 2 * 1024 * 1024,
        on_shutdown: Optional[Any] = None,
    ):
        self.relay = relay
        self.max_message_size = max_message_size
        self.on_shutdown = on_shutdown
        self.actions = 0

    def cleanup(self) -> None:
        """Called by ``appfl.comm.grpc.serve`` on shutdown. Nothing to release."""
        if self.on_shutdown is not None:
            self.on_shutdown(self.actions)

    def InvokeCustomAction(self, request_iterator, context):
        request = CustomActionRequest()
        received = b""
        for chunk in request_iterator:
            received += chunk.data_bytes
        request.ParseFromString(received)

        meta_data = yaml.safe_load(request.meta_data) if request.meta_data else {}
        # The client puts its agent id in the metadata; fall back to the gRPC client id so a
        # federation whose site names match its client ids needs no extra wiring.
        agent_id = meta_data.pop("agent_id", request.header.client_id)
        results = self.relay.handle(request.action, agent_id, **meta_data)
        self.actions += 1

        response = CustomActionResponse(
            header=ServerHeader(status=ServerStatus.RUN),
            results=yaml.dump(results),
        )
        yield from proto_to_databuffer(response, max_message_size=self.max_message_size)
