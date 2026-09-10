from typing import Optional, List
from appfl.logger import ServerAgentFileLogger
from appfl.comm.base import ServerDrivenCommunicator
from appfl.config import ClientAgentConfig, ServerAgentConfig
from .tes_backend import TESBackend


class TESServerCommunicator(ServerDrivenCommunicator):
    """
    GA4GH Task Execution Service (TES) server communicator for APPFL.

    This is a thin wrapper over :class:`~appfl.comm.base.ServerDrivenCommunicator`
    that routes *all* clients through a single :class:`TESBackend`. The generic
    driver owns task bookkeeping and the receive loop, while the backend
    implements the TES transport (submitting federated learning tasks to
    GA4GH TES-compliant compute infrastructures).
    """

    def __init__(
        self,
        server_agent_config: ServerAgentConfig,
        client_agent_configs: List[ClientAgentConfig],
        logger: Optional[ServerAgentFileLogger] = None,
        **kwargs,
    ):
        self.comm_type = "tes"
        super().__init__(
            server_agent_config=server_agent_config,
            client_agent_configs=client_agent_configs,
            logger=logger,
            **kwargs,
        )

    def _build_backends(self, **kwargs):
        self._add_backend(
            TESBackend(
                server_agent_config=self.server_agent_config,
                client_agent_configs=self.client_agent_configs,
                experiment_id=self.experiment_id,
                logger=self.logger,
                **kwargs,
            )
        )
