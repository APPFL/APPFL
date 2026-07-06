import uuid
import warnings
from omegaconf import OmegaConf
from globus_sdk.scopes import AuthScopes, ComputeScopes
from globus_sdk import AccessTokenAuthorizer
from globus_compute_sdk import Executor, Client
from typing import Optional, Dict, List, Union, OrderedDict, Tuple, Any
from appfl.comm.utils.s3_utils import send_model_by_s3
from appfl.logger import ServerAgentFileLogger
from appfl.comm.base import ServerCommBackend, ModelTransferHelper
from appfl.comm.utils.s3_storage import CloudStorage
from appfl.config import ClientAgentConfig, ServerAgentConfig
from .utils.endpoint import GlobusComputeClientEndpoint
from globus_compute_sdk.serialize import CombinedCode
from globus_compute_sdk.sdk.login_manager import AuthorizerLoginManager


class GlobusComputeBackend(ServerCommBackend):
    """
    Server communication backend that uses Globus Compute for orchestrating
    federated learning tasks on remote client endpoints.

    Globus Compute is a distributed function-as-a-service platform that allows
    users to run functions on specified remote endpoints. For more details, check
    the Globus Compute SDK documentation at
    https://globus-compute.readthedocs.io/en/latest/endpoints.html.

    :param `server_agent_config`: The server agent configuration.
    :param `client_agent_configs`: The subset of client configurations served via Globus Compute.
    :param `experiment_id`: Shared experiment id assigned by the driver.
    :param [Optional] `logger`: Logger object shared with the driver.
    """

    def __init__(
        self,
        server_agent_config: ServerAgentConfig,
        client_agent_configs: List[ClientAgentConfig],
        experiment_id: str,
        logger: Optional[ServerAgentFileLogger] = None,
        **kwargs,
    ):
        super().__init__(
            comm_type="globus_compute",
            server_agent_config=server_agent_config,
            client_agent_configs=client_agent_configs,
            experiment_id=experiment_id,
            logger=logger,
            **kwargs,
        )
        # Model-transfer storage (S3 / ProxyStore) is owned by the backend.
        self.use_s3bucket = ModelTransferHelper.init_s3(
            server_agent_config, self.logger, self.comm_type, self.experiment_id
        )
        self.use_proxystore, self.proxystore = ModelTransferHelper.init_proxystore(
            server_agent_config, self.logger
        )
        assert not (self.use_proxystore and self.use_s3bucket), (
            "Proxystore and S3 bucket cannot be used together."
        )

        self._load_gce(**kwargs)

        # Initiate the Globus Compute client endpoints.
        self.client_endpoints: Dict[str, GlobusComputeClientEndpoint] = {}
        _client_id_check_set = set()
        for client_config in client_agent_configs:
            assert hasattr(client_config, "endpoint_id"), (
                "Client configuration must have an endpoint_id."
            )
            # Read the client dataloader source file
            with open(client_config.data_configs.dataset_path) as file:
                client_config.data_configs.dataset_source = file.read()
            del client_config.data_configs.dataset_path
            client_id = str(
                client_config.client_id
                if hasattr(client_config, "client_id")
                else (
                    client_config.train_configs.logging_id
                    if (
                        hasattr(client_config, "train_configs")
                        and hasattr(client_config.train_configs, "logging_id")
                    )
                    else client_config.endpoint_id
                )
            )
            assert client_id not in _client_id_check_set, (
                f"Client ID {client_id} is not unique for this client configuration.\n{client_config}"
            )
            _client_id_check_set.add(client_id)
            client_endpoint_id = client_config.endpoint_id
            client_config.experiment_id = self.experiment_id
            client_config.comm_type = self.comm_type
            # Raise deprecation warning for logging_id
            if hasattr(client_config.train_configs, "logging_id"):
                warnings.warn(
                    "client_agent_config.train_configs.logging_id is deprecated. Please use client_id instead.",
                    DeprecationWarning,
                )
            # logging information regarding wandb
            if hasattr(
                client_config, "wandb_configs"
            ) and client_config.wandb_configs.get("enable_wandb", False):
                self.logger.info(f"{client_id} is using wandb for logging. ")

            self.client_endpoints[client_id] = GlobusComputeClientEndpoint(
                client_id=client_id,
                client_endpoint_id=client_endpoint_id,
                client_config=OmegaConf.merge(
                    server_agent_config.client_configs, client_config
                ),
            )

        # Map each in-flight future to its client id so the client endpoint can be
        # reset to AVAILABLE once its result has been processed (see process_result).
        self._fut_client: Dict[Any, str] = {}

        # Log initialization with multi-endpoint support
        self.logger.info(
            f"Backend manages {len(self.client_endpoints)} client endpoint(s)."
        )
        for client_id, endpoint in self.client_endpoints.items():
            self.logger.info(
                f"Client '{client_id}' -> {endpoint.client_endpoint_id}"
            )

    @property
    def client_ids(self) -> List[str]:
        return list(self.client_endpoints.keys())

    def prepare_model(
        self, model: Optional[Union[Dict, OrderedDict, bytes]]
    ) -> Any:
        """Upload/proxy the shared model once before submitting it to clients."""
        if model is None:
            return None
        if self.use_s3bucket:
            return send_model_by_s3(
                self.experiment_id, self.comm_type, model, "server"
            )
        elif self.use_proxystore:
            return self.proxystore.proxy(model)
        return model

    def submit_task(
        self,
        client_id: str,
        task_name: str,
        prepared_model: Any = None,
        metadata: Optional[Dict] = None,
        need_model_response: bool = False,
    ) -> Tuple[str, Any]:
        client_metadata = dict(metadata) if metadata else {}
        if need_model_response and self.use_s3bucket:
            local_model_key = f"{str(uuid.uuid4())}_client_state_{client_id}"
            local_model_url = CloudStorage.presign_upload_object(local_model_key)
            client_metadata["local_model_key"] = local_model_key
            client_metadata["local_model_url"] = local_model_url
        task_id, task_future = self.client_endpoints[client_id].submit_task(
            self.gce,
            task_name,
            prepared_model,
            client_metadata,
        )
        self._fut_client[task_future] = client_id
        return task_id, task_future

    def process_result(self, future: Any) -> Tuple[Any, Dict]:
        result = future.result()
        model, metadata = ModelTransferHelper.parse_result(result, self.use_s3bucket)
        # Reading the endpoint status resets it to AVAILABLE now that the task is
        # done, so the endpoint can accept the next task submission.
        client_id = self._fut_client.pop(future, None)
        if client_id is not None:
            self.client_endpoints[client_id].status
        return model, metadata

    def cancel_task(self, task_id: str, future: Any, client_id: str):
        future.cancel()
        self._fut_client.pop(future, None)
        self.client_endpoints[client_id].cancel_task()

    def shutdown(self):
        self.gce.shutdown(wait=False, cancel_futures=True)
        # Clean-up cloud storage
        if self.use_s3bucket:
            CloudStorage.clean_up()
            self.logger.warning('[debug] S3 bucket cleanup complete.')
        # Clean-up proxystore
        if hasattr(self, "proxystore") and self.proxystore is not None:
            try:
                self.proxystore.close(clear=True)
            except:  # noqa: E722
                self.proxystore.close()
        self.logger.info("Backend shutdown complete.")

    def _load_gce(self, **kwargs):
        """
        Load the Globus Compute Executor.
        """
        # Assert compute_token and openid_token are both provided if necessary
        assert ("compute_token" in kwargs and "openid_token" in kwargs) or (
            "compute_token" not in kwargs and "openid_token" not in kwargs
        ), (
            "Both compute_token and openid_token must be provided if one of them is provided."
        )

        if "compute_token" in kwargs and "openid_token" in kwargs:
            compute_login_manager = AuthorizerLoginManager(
                authorizers={
                    ComputeScopes.resource_server: AccessTokenAuthorizer(
                        kwargs["compute_token"]
                    ),
                    AuthScopes.resource_server: AccessTokenAuthorizer(
                        kwargs["openid_token"]
                    ),
                }
            )
            compute_login_manager.ensure_logged_in()
            gcc = Client(
                login_manager=compute_login_manager,
                code_serialization_strategy=CombinedCode(),
            )
        else:
            gcc = Client(
                code_serialization_strategy=CombinedCode(),
            )
        self.gce = Executor(client=gcc)  # Globus Compute Executor
