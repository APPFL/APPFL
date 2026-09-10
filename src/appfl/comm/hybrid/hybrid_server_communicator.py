from typing import Optional, List, Dict
from appfl.logger import ServerAgentFileLogger
from appfl.comm.base import ServerDrivenCommunicator
from appfl.config import ClientAgentConfig, ServerAgentConfig
from appfl.comm.tes.tes_backend import TESBackend
from appfl.comm.globus_compute.globus_compute_backend import GlobusComputeBackend

# Registry mapping a client's ``comm_type`` to its server-side backend class.
# Add new server-driven transports here to make them usable in a hybrid federation.
_BACKEND_REGISTRY = {
    "globus_compute": GlobusComputeBackend,
    "tes": TESBackend,
}


class HybridServerCommunicator(ServerDrivenCommunicator):
    """
    A *hybrid* server-driven communicator: a single federation whose clients are
    reached through different transports (e.g., some via Globus Compute and some
    via TES).

    Each client's transport is resolved from its configuration and the clients are
    grouped by transport; one :class:`~appfl.comm.base.ServerCommBackend` is
    constructed per transport. The generic
    :class:`~appfl.comm.base.ServerDrivenCommunicator` then drives all backends
    through a single unified receive loop, which works because every backend
    returns ``concurrent.futures.Future``-compatible handles.

    Transport resolution for a client (in order):
    1. Explicit ``comm_configs.comm_type`` (``"globus_compute"`` or ``"tes"``).
    2. Auto-inference: ``endpoint_id`` present -> Globus Compute;
       ``comm_configs.tes_configs`` present -> TES.

    Transport-specific keyword arguments (e.g., Globus Compute ``compute_token`` /
    ``openid_token``, or TES ``auth_token``) are forwarded to every backend; each
    backend only consumes the keys it recognizes.

    :param `server_agent_config`: The server agent configuration.
    :param `client_agent_configs`: A list of client agent configurations.
    :param [Optional] `logger`: Optional logger object.
    """

    def __init__(
        self,
        server_agent_config: ServerAgentConfig,
        client_agent_configs: List[ClientAgentConfig],
        logger: Optional[ServerAgentFileLogger] = None,
        **kwargs,
    ):
        self.comm_type = "hybrid"
        super().__init__(
            server_agent_config=server_agent_config,
            client_agent_configs=client_agent_configs,
            logger=logger,
            **kwargs,
        )

    def _build_backends(self, **kwargs):
        # Group client configs by resolved transport, preserving order.
        grouped: Dict[str, List[ClientAgentConfig]] = {}
        for client_config in self.client_agent_configs:
            comm_type = self._resolve_comm_type(client_config, self.logger)
            grouped.setdefault(comm_type, []).append(client_config)

        for comm_type, client_configs in grouped.items():
            backend_cls = _BACKEND_REGISTRY[comm_type]
            self._add_backend(
                backend_cls(
                    server_agent_config=self.server_agent_config,
                    client_agent_configs=client_configs,
                    experiment_id=self.experiment_id,
                    logger=self.logger,
                    **kwargs,
                )
            )
        self.logger.info(
            f"Hybrid communicator manages {len(self.client_ids)} clients across "
            f"transports: {sorted(grouped.keys())}."
        )

    @staticmethod
    def _resolve_comm_type(client_config: ClientAgentConfig, logger=None) -> str:
        """Resolve the transport (``comm_type``) for a single client configuration."""
        comm_configs = getattr(client_config, "comm_configs", None)
        client_id = getattr(client_config, "client_id", "<unknown>")
        # 1. Explicit comm_type.
        if comm_configs is not None and hasattr(comm_configs, "comm_type"):
            comm_type = str(comm_configs.comm_type)
            if comm_type not in _BACKEND_REGISTRY:
                raise ValueError(
                    f"Unsupported comm_type '{comm_type}' for a hybrid client. "
                    f"Supported: {sorted(_BACKEND_REGISTRY.keys())}."
                )
            return comm_type
        # 2. Auto-inference (best-effort; explicit comm_type is recommended).
        inferred = None
        if hasattr(client_config, "endpoint_id"):
            inferred = "globus_compute"
        elif comm_configs is not None and hasattr(comm_configs, "tes_configs"):
            inferred = "tes"
        if inferred is not None:
            if logger is not None:
                logger.warning(
                    f"comm_type for client '{client_id}' was not specified and has "
                    f"been auto-inferred as '{inferred}'. It is recommended to set "
                    f"'comm_configs.comm_type' explicitly in the client config to "
                    f"avoid ambiguity."
                )
            return inferred
        raise ValueError(
            f"Cannot determine comm_type for client '{client_id}'. Set "
            f"comm_configs.comm_type explicitly, or provide an 'endpoint_id' "
            f"(Globus Compute) or 'comm_configs.tes_configs' (TES)."
        )
