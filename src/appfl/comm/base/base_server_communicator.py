import time
import logging
from datetime import datetime
from abc import abstractmethod
from omegaconf import OmegaConf
from appfl.comm.utils.config import ClientTask
from appfl.logger import ServerAgentFileLogger
from appfl.comm.base.model_transfer_helper import ModelTransferHelper
from appfl.config import ClientAgentConfig, ServerAgentConfig
from typing import List, Optional, Union, Dict, OrderedDict, Tuple, Any


class BaseServerCommunicator:
    def __init__(
        self,
        server_agent_config: ServerAgentConfig,
        client_agent_configs: List[ClientAgentConfig],
        logger: Optional[ServerAgentFileLogger] = None,
        **kwargs,
    ):
        self.server_agent_config = server_agent_config
        self.client_agent_configs = client_agent_configs
        self.logger = logger if logger is not None else self._default_logger()
        self.experiment_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self._sanity_check()
        self._init_storage(server_agent_config)
        self.executing_tasks: Dict[str, ClientTask] = {}
        self.executing_task_futs: Dict[Any, str] = {}

    def _init_storage(self, server_agent_config):
        """
        Initialize the model-transfer storage backends (AWS S3 and ProxyStore).

        This is a hook so that subclasses which delegate model transfer to
        per-client backends (e.g., the ``ServerDrivenCommunicator``) can override
        it to a no-op and let each backend own its own storage.
        """
        self.use_s3bucket = ModelTransferHelper.init_s3(
            server_agent_config, self.logger, self.comm_type, self.experiment_id
        )
        self.use_proxystore, self.proxystore = ModelTransferHelper.init_proxystore(
            server_agent_config, self.logger
        )
        assert not (self.use_proxystore and self.use_s3bucket), (
            "Proxystore and S3 bucket cannot be used together."
        )

    @abstractmethod
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
        pass

    @abstractmethod
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
        Send a specific task to one specific client.
        :param `client_id`: The client id to which the task is sent.
        :param `task_name`: Name of the task to be executed on the clients
        :param [Optional] `model`: Model to be sent to the clients
        :param [Optional] `metadata`: Additional metadata to be sent to the clients
        :param `need_model_response`: Whether the task requires a model response from the clients
            If so, the server will provide a pre-signed URL for the clients to upload the model if using S3.
        """

    @abstractmethod
    def recv_result_from_all_clients(self) -> Tuple[Dict, Dict]:
        """
        Receive task results from all clients that have running tasks.
        :return `client_results`: A dictionary containing the results from all clients - Dict[client_id, client_model]
        :return `client_metadata`: A dictionary containing the metadata from all clients - Dict[client_id, client_metadata]
        """
        pass

    @abstractmethod
    def recv_result_from_one_client(self) -> Tuple[str, Any, Dict]:
        """
        Receive task results from the first client that finishes the task.
        :return `client_id`: The client id from which the result is received.
        :return `client_model`: The model returned from the client
        :return `client_metadata`: The metadata returned from the client
        """
        pass

    @abstractmethod
    def shutdown_all_clients(self):
        """Cancel all the running tasks on the clients and shutdown the globus compute executor."""
        pass

    @abstractmethod
    def cancel_all_tasks(self):
        """Cancel all on-the-fly client tasks."""
        pass

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

    def _parse_result(self, result):
        """Parse a client result, downloading the model from S3/ProxyStore if needed."""
        return ModelTransferHelper.parse_result(result, self.use_s3bucket)

    def _register_task(self, task_id, task_fut, client_id, task_name):
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

    def _sanity_check(self):
        # Sanity check for number of clients
        num_clients = (
            self.server_agent_config.server_configs.num_clients
            if hasattr(self.server_agent_config.server_configs, "num_clients")
            else self.server_agent_config.server_configs.scheduler_kwargs.num_clients
            if (
                hasattr(self.server_agent_config.server_configs, "scheduler_kwargs")
                and hasattr(
                    self.server_agent_config.server_configs.scheduler_kwargs,
                    "num_clients",
                )
            )
            else self.server_agent_config.server_configs.aggregator_kwargs.num_clients
        )
        assert num_clients == len(self.client_agent_configs), (
            "Number of clients in the server configuration does not match the number of client configurations."
        )

    def _check_deprecation(
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
