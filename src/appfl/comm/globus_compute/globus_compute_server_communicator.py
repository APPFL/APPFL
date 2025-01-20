import os
import time
import uuid
import pathlib
import logging
import warnings
from datetime import datetime
from omegaconf import OmegaConf
from collections import OrderedDict
from globus_sdk.scopes import AuthScopes
from globus_sdk import AccessTokenAuthorizer
from proxystore.store import Store
from proxystore.proxy import Proxy, extract
from globus_compute_sdk import Executor, Client
from concurrent.futures import Future, as_completed
from typing import Optional, Dict, List, Union, Tuple, Any
from appfl.logger import ServerAgentFileLogger
from appfl.misc.utils import get_proxystore_connector
from appfl.config import ClientAgentConfig, ServerAgentConfig
from .utils.config import ClientTask
from .utils.endpoint import GlobusComputeClientEndpoint
from .utils.s3_storage import CloudStorage, LargeObjectWrapper
from globus_compute_sdk.serialize import CombinedCode
from globus_compute_sdk.sdk.login_manager import AuthorizerLoginManager
from globus_compute_sdk.sdk.login_manager.manager import ComputeScopeBuilder


class GlobusComputeServerCommunicator:
    """
    Communicator used by the federated learning server which plans to use Globus Compute
    for orchestrating the federated learning experiments.

    Globus Compute is a distributed function-as-a-service platform that allows users to run
    functions on specified remote endpoints. For more details, check the Globus Compute SDK
    documentation at https://globus-compute.readthedocs.io/en/latest/endpoints.html.

    :param `gcc`: Globus Compute client object
    :param `server_agent_config`: The server agent configuration
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
        # Assert compute_token and openid_token are both provided if necessary
        assert ("compute_token" in kwargs and "openid_token" in kwargs) or (
            "compute_token" not in kwargs and "openid_token" not in kwargs
        ), (
            "Both compute_token and openid_token must be provided if one of them is provided."
        )

        if "compute_token" in kwargs and "openid_token" in kwargs:
            ComputeScopes = ComputeScopeBuilder()
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
        self.logger = logger if logger is not None else self._default_logger()
        # Sanity check for configurations: Check for the number of clients
        num_clients = (
            server_agent_config.server_configs.num_clients
            if hasattr(server_agent_config.server_configs, "num_clients")
            else server_agent_config.server_configs.scheduler_kwargs.num_clients
            if (
                hasattr(server_agent_config.server_configs, "scheduler_kwargs")
                and hasattr(
                    server_agent_config.server_configs.scheduler_kwargs, "num_clients"
                )
            )
            else server_agent_config.server_configs.aggregator_kwargs.num_clients
        )
        assert num_clients == len(client_agent_configs), (
            "Number of clients in the server configuration does not match the number of client configurations."
        )
        client_config_from_server = server_agent_config.client_configs
        # Create a unique experiment ID for this federated learning experiment
        experiment_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
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
            client_config.experiment_id = experiment_id
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
                client_config=OmegaConf.merge(client_config_from_server, client_config),
            )
        # Initialize the S3 bucket for large model transfer if necessary.
        if hasattr(server_agent_config.server_configs, "comm_configs") and hasattr(
            server_agent_config.server_configs.comm_configs, "globus_compute_configs"
        ):
            s3_bucket = server_agent_config.server_configs.comm_configs.globus_compute_configs.get(
                "s3_bucket", None
            )
        else:
            s3_bucket = None
        self.use_s3bucket = s3_bucket is not None
        if self.use_s3bucket:
            self.logger.info(f"Using S3 bucket {s3_bucket} for model transfer.")
            s3_creds_file = server_agent_config.server_configs.comm_configs.globus_compute_configs.get(
                "s3_creds_file", None
            )
            s3_temp_dir = server_agent_config.server_configs.comm_configs.globus_compute_configs.get(
                "s3_temp_dir",
                str(
                    pathlib.Path.home()
                    / ".appfl"
                    / "globus_compute"
                    / "server"
                    / experiment_id
                ),
            )
            if not os.path.exists(s3_temp_dir):
                pathlib.Path(s3_temp_dir).mkdir(parents=True, exist_ok=True)
            CloudStorage.init(s3_bucket, s3_creds_file, s3_temp_dir, self.logger)

        # Load proxystore
        self._load_proxystore(server_agent_config)
        assert not (self.use_proxystore and self.use_s3bucket), (
            "Proxystore and S3 bucket cannot be used together."
        )

        self.executing_tasks: Dict[str, ClientTask] = {}
        self.executing_task_futs: Dict[Future, str] = {}

    def send_task_to_all_clients(
        self,
        task_name: str,
        *,
        model: Optional[Union[Dict, OrderedDict, bytes]] = None,
        metadata: Union[Dict, List[Dict]] = {},
        need_model_response: bool = False,
    ):
        """
        Send a specific task to all clients.
        :param `task_name`: Name of the task to be executed on the clients
        :param [Optional] `model`: Model to be sent to the clients
        :param [Optional] `metadata`: Additional metadata to be sent to the clients
        :param `need_model_response`: Whether the task requires a model response from the clients
            If so, the server will provide a pre-signed URL for the clients to upload the model if using S3.
        """
        if self.use_s3bucket and model is not None:
            model_wrapper = LargeObjectWrapper(
                data=model,
                name=str(uuid.uuid4()) + "_server_state",
            )
            if not model_wrapper.can_send_directly:
                model = CloudStorage.upload_object(
                    model_wrapper, register_for_clean=True
                )
        elif self.use_proxystore and model is not None:
            model = self.proxystore.proxy(model)
        for i, client_id in enumerate(self.client_endpoints):
            client_metadata = metadata[i] if isinstance(metadata, list) else metadata
            if need_model_response and self.use_s3bucket:
                local_model_key = f"{str(uuid.uuid4())}_client_state_{client_id}"
                local_model_url = CloudStorage.presign_upload_object(local_model_key)
                client_metadata["local_model_key"] = local_model_key
                client_metadata["local_model_url"] = local_model_url
            task_id, task_future = self.client_endpoints[client_id].submit_task(
                self.gce,
                task_name,
                model,
                client_metadata,
            )
            self.__register_task(task_id, task_future, client_id, task_name)
            self.logger.info(f"Task '{task_name}' is assigned to {client_id}.")

    def send_task_to_one_client(
        self,
        client_id: str,
        task_name: str,
        *,
        model: Optional[Union[Dict, OrderedDict, bytes]] = None,
        metadata: Optional[Dict] = {},
        need_model_response: bool = False,
    ):
        """
        Send a specific task to one specific client endpoint.
        :param `client_id`: The client id to which the task is sent.
        :param `task_name`: Name of the task to be executed on the clients
        :param [Optional] `model`: Model to be sent to the clients
        :param [Optional] `metadata`: Additional metadata to be sent to the clients
        :param `need_model_response`: Whether the task requires a model response from the clients
            If so, the server will provide a pre-signed URL for the clients to upload the model if using S3.
        """
        if self.use_s3bucket and model is not None:
            model_wrapper = LargeObjectWrapper(
                data=model,
                name=str(uuid.uuid4()) + "_server_state",
            )
            if not model_wrapper.can_send_directly:
                model = CloudStorage.upload_object(
                    model_wrapper, register_for_clean=True
                )
        elif self.use_proxystore and model is not None:
            model = self.proxystore.proxy(model)
        if need_model_response and self.use_s3bucket:
            local_model_key = f"{str(uuid.uuid4())}_client_state_{client_id}"
            local_model_url = CloudStorage.presign_upload_object(local_model_key)
            metadata["local_model_key"] = local_model_key
            metadata["local_model_url"] = local_model_url
        task_id, task_future = self.client_endpoints[client_id].submit_task(
            self.gce,
            task_name,
            model,
            metadata,
        )
        self.__register_task(task_id, task_future, client_id, task_name)
        self.logger.info(f"Task '{task_name}' is assigned to {client_id}.")

    def recv_result_from_all_clients(self) -> Tuple[Dict, Dict]:
        """
        Receive task results from all clients that have running tasks.
        :return `client_results`: A dictionary containing the results from all clients - Dict[client_id, client_model]
        :return `client_metadata`: A dictionary containing the metadata from all clients - Dict[client_id, client_metadata]
        """
        client_results, client_metadata = {}, {}
        while len(self.executing_task_futs):
            fut = next(as_completed(list(self.executing_task_futs)))
            task_id = self.executing_task_futs[fut]
            client_id = self.executing_tasks[task_id].client_id
            try:
                result = fut.result()
                client_model, client_metadata_local = (
                    self.__parse_globus_compute_result(result)
                )
                client_metadata_local = self.__check_deprecation(
                    client_id, client_metadata_local
                )
                client_results[client_id] = client_model
                client_metadata[client_id] = client_metadata_local
                # Set the status of the finished task
                client_log = client_metadata_local.get("log", {})
                self.executing_tasks[task_id].end_time = time.time()
                self.executing_tasks[task_id].success = True
                self.executing_tasks[task_id].log = client_log  # TODO: Check this line
                # Clean up the task
                self.logger.info(
                    f"Received results of task '{self.executing_tasks[task_id].task_name}' from {client_id}."
                )
                self.client_endpoints[client_id].status
                self.executing_tasks.pop(task_id)
                self.executing_task_futs.pop(fut)
            except Exception as e:
                self.logger.error(
                    f"Task {self.executing_tasks[task_id].task_name} on {client_id} failed with an error."
                )
                raise e
        return client_results, client_metadata

    def recv_result_from_one_client(self) -> Tuple[str, Any, Dict]:
        """
        Receive task results from the first client that finishes the task.
        :return `client_id`: The client endpoint id from which the result is received.
        :return `client_model`: The model returned from the client
        :return `client_metadata`: The metadata returned from the client
        """
        assert len(self.executing_task_futs), (
            "There is no active client endpoint running tasks."
        )
        try:
            fut = next(as_completed(list(self.executing_task_futs)))
            task_id = self.executing_task_futs[fut]
            result = fut.result()
            client_id = self.executing_tasks[task_id].client_id
            client_model, client_metadata = self.__parse_globus_compute_result(result)
            client_metadata = self.__check_deprecation(client_id, client_metadata)
            # Set the status of the finished task
            client_log = client_metadata.get("log", {})
            self.executing_tasks[task_id].end_time = time.time()
            self.executing_tasks[task_id].success = True
            self.executing_tasks[task_id].log = client_log  # TODO: Check this line
            # Clean up the task
            self.logger.info(
                f"Received results of task '{self.executing_tasks[task_id].task_name}' from {client_id}."
            )
            self.client_endpoints[client_id].status
            self.executing_tasks.pop(task_id)
            self.executing_task_futs.pop(fut)
        except Exception as e:
            client_id = self.executing_tasks[task_id].client_id
            self.logger.error(
                f"Task {self.executing_tasks[task_id].task_name} on {client_id} failed with an error."
            )
            raise e
        return client_id, client_model, client_metadata

    def shutdown_all_clients(self):
        """Cancel all the running tasks on the clients and shutdown the globus compute executor."""
        self.logger.info("Shutting down all clients......")
        self.gce.shutdown(wait=False, cancel_futures=True)
        # Clean-up cloud storage
        if self.use_s3bucket:
            CloudStorage.clean_up()
        # Clean-up proxystore
        if hasattr(self, "proxystore") and self.proxystore is not None:
            try:
                self.proxystore.close(clear=True)
            except:  # noqa: E722
                self.proxystore.close()
        self.logger.info(
            "The server and all clients have been shutted down successfully."
        )

    def cancel_all_tasks(self):
        """Cancel all on-the-fly client tasks."""
        for task_fut in self.executing_task_futs:
            task_fut.cancel()
            task_id = self.executing_task_futs[task_fut]
            client_id = self.executing_tasks[task_id].client_id
            self.client_endpoints[client_id].cancel_task()
        self.executing_task_futs = {}
        self.executing_tasks = {}

    def __check_deprecation(
        self,
        client_id: str,
        client_metadata: Dict,
    ):
        """
        This function is used to check deprecation on the client site packages.
        """
        if not hasattr(self, "_version_deprecation_warning_set"):
            self._version_deprecation_warning_set = set()
        if "_deprecated" in client_metadata:
            if client_id not in self._version_deprecation_warning_set:
                self.logger.warning(
                    f"{client_id} is using a deprecated version of appfl, and it is highly recommended to update it to at least version 1.2.1."
                )
                self._version_deprecation_warning_set.add(client_id)
            client_metadata.pop("_deprecated")
        return client_metadata

    def __parse_globus_compute_result(self, result):
        """
        Parse the returned results from a Globus Compute endpoint.
        The results can be composed of two parts:
        - Model parameters (can be model, gradients, compressed model, etc.)
        - Metadata (may contain additional information such as logs, etc.)
        :param `result`: The result returned from the Globus Compute endpoint.
        :return `model`: The model parameters returned from the client
        :return `metadata`: The metadata returned from the client
        """
        if isinstance(result, tuple):
            model, metadata = result
        else:
            model, metadata = result, {}
        # Download model from S3 bucket or ProxyStore if necessary
        if isinstance(model, Proxy):
            model = extract(model)
        if self.use_s3bucket:
            if CloudStorage.is_cloud_storage_object(model):
                model = CloudStorage.download_object(
                    model, delete_cloud=True, delete_local=True
                )
        return model, metadata

    def __register_task(self, task_id, task_fut, client_id, task_name):
        """
        Register new client task to the list of executing tasks - call after task submission.
        """
        self.executing_tasks[task_id] = OmegaConf.structured(
            ClientTask(
                task_id=task_id,
                task_name=task_name,
                client_id=client_id,
                start_time=time.time(),
            )
        )
        self.executing_task_futs[task_fut] = task_id

    def _default_logger(self):
        """Create a default logger for the gRPC server if no logger provided."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("[%(asctime)s %(levelname)-4s server]: %(message)s")
        s_handler = logging.StreamHandler()
        s_handler.setLevel(logging.INFO)
        s_handler.setFormatter(fmt)
        logger.addHandler(s_handler)
        return logger

    def _load_proxystore(self, server_agent_config) -> None:
        """
        Create the proxystore for storing and sending model parameters from the server to the clients.
        """
        self.proxystore = None
        self.use_proxystore = False
        if (
            hasattr(server_agent_config.server_configs, "comm_configs")
            and hasattr(
                server_agent_config.server_configs.comm_configs, "proxystore_configs"
            )
            and server_agent_config.server_configs.comm_configs.proxystore_configs.get(
                "enable_proxystore", False
            )
        ):
            self.use_proxystore = True
            self.proxystore = Store(
                name="server-proxystore",
                connector=get_proxystore_connector(
                    server_agent_config.server_configs.comm_configs.proxystore_configs.connector_type,
                    server_agent_config.server_configs.comm_configs.proxystore_configs.connector_configs,
                ),
            )
            self.logger.info(
                f"Server using proxystore for model transfer with store: {server_agent_config.server_configs.comm_configs.proxystore_configs.connector_type}."
            )
