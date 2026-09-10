from typing import Optional, List
from appfl.logger import ServerAgentFileLogger
from appfl.comm.base import ServerDrivenCommunicator
from appfl.config import ClientAgentConfig, ServerAgentConfig
from .globus_compute_backend import GlobusComputeBackend


class GlobusComputeServerCommunicator(ServerDrivenCommunicator):
    """
    Communicator used by the federated learning server which plans to use Globus Compute
    for orchestrating the federated learning experiments.

    This is a thin wrapper over :class:`~appfl.comm.base.ServerDrivenCommunicator`
    that routes *all* clients through a single :class:`GlobusComputeBackend`. The
    generic driver owns task bookkeeping and the receive loop, while the backend
    implements the Globus Compute transport.

    Globus Compute is a distributed function-as-a-service platform that allows users to run
    functions on specified remote endpoints. For more details, check the Globus Compute SDK
    documentation at https://globus-compute.readthedocs.io/en/latest/endpoints.html.

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
        self.comm_type = "globus_compute"
        super().__init__(
            server_agent_config=server_agent_config,
            client_agent_configs=client_agent_configs,
            logger=logger,
            **kwargs,
        )

    def _build_backends(self, **kwargs):
        self._add_backend(
            GlobusComputeBackend(
                server_agent_config=self.server_agent_config,
                client_agent_configs=self.client_agent_configs,
                experiment_id=self.experiment_id,
                logger=self.logger,
                **kwargs,
            )
        )
