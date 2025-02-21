import uuid
import time
from typing import List, Optional, Union, Dict, OrderedDict, Any, Tuple

from omegaconf import OmegaConf

from appfl.comm.base.base_server_communicator import BaseServerCommunicator
from appfl.comm.ray.ray_client_communicator import RayClientCommunicator
from appfl.comm.utils.s3_storage import CloudStorage
from appfl.config import ServerAgentConfig, ClientAgentConfig
from appfl.logger import ServerAgentFileLogger
from appfl.comm.utils.config import ClientTask
import ray


class RayServerCommunicator(BaseServerCommunicator):
    def __init__(
        self,
        server_agent_config: ServerAgentConfig,
        client_agent_configs: List[ClientAgentConfig],
        logger: Optional[ServerAgentFileLogger] = None,
        **kwargs,
    ):
        self.comm_type = "ray"
        super().__init__(server_agent_config, client_agent_configs, logger, **kwargs)
        ray.init(address="auto")
        self.client_actors = {}
        for client_config in client_agent_configs:
            endpoint_id = client_config["endpoint_id"]
            client_config.experiment_id = self.experiment_id
            client_config = OmegaConf.merge(
                server_agent_config.client_configs, client_config
            )
            client_config.comm_configs.comm_type = self.comm_type
            if (
                hasattr(server_agent_config.client_configs, "comm_configs")
                and hasattr(
                    server_agent_config.client_configs.comm_configs, "ray_config"
                )
                and not server_agent_config.client_configs.comm_configs.ray_config.get(
                    "assign_random", True
                )
            ):
                self.client_actors[endpoint_id] = RayClientCommunicator.options(
                    resources={client_config["client_id"]: 1}
                ).remote(server_agent_config, client_config)
            else:
                self.client_actors[endpoint_id] = RayClientCommunicator.remote(
                    server_agent_config, client_config
                )

        self.executing_tasks: Dict[str, ClientTask] = {}
        self.executing_task_futs: Dict[Any, str] = {}

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
        for i, client_id in enumerate(self.client_actors):
            client_metadata = metadata[i] if isinstance(metadata, list) else metadata
            task_id, task_ref = self.__send_task(
                self.client_actors[client_id], task_name, model, client_metadata
            )
            super()._register_task(task_id, task_ref, client_id, task_name)
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
        task_id, task_ref = self.__send_task(
            self.client_actors[client_id], task_name, model, metadata
        )
        self._register_task(task_id, task_ref, client_id, task_name)
        self.logger.info(f"Task '{task_name}' is assigned to {client_id}.")

    def recv_result_from_all_clients(self) -> Tuple[Dict, Dict]:
        """
        Receive task results from all clients that have running tasks.
        :return `client_results`: A dictionary containing the results from all clients - Dict[client_id, client_model]
        :return `client_metadata`: A dictionary containing the metadata from all clients - Dict[client_id, client_metadata]
        """
        client_results, client_metadata = {}, {}
        while len(self.executing_task_futs):
            done_futs, _ = ray.wait(
                list(self.executing_task_futs.keys()), num_returns=1
            )

            if not done_futs:
                continue

            fut = done_futs[0]
            task_id = self.executing_task_futs[fut]
            client_id = self.executing_tasks[task_id].client_id

            try:
                result = ray.get(fut)
                client_model, client_metadata_local = self._parse_result(result)
                client_results[client_id] = client_model
                client_metadata[client_id] = client_metadata_local
                self.__update_executing_task(
                    client_metadata_local, task_id, client_id, fut
                )

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
        ready_refs, _ = ray.wait(
            list(self.executing_task_futs), num_returns=1, timeout=None
        )
        finished_ref = ready_refs[0]
        task_id = self.executing_task_futs[finished_ref]
        try:
            result = ray.get(finished_ref)
            client_id = self.executing_tasks[task_id].client_id
            client_model, client_metadata = self._parse_result(result)
            self.__update_executing_task(
                client_metadata, task_id, client_id, finished_ref
            )
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
        ray.shutdown()
        # Clean-up cloud storage
        if self.use_s3bucket:
            CloudStorage.clean_up()
        # Clean-up proxystore
        if hasattr(self, "proxystore") and self.proxystore is not None:
            try:
                self.proxystore.close(clear=True)
            except Exception:  # noqa: E722
                self.proxystore.close()
        self.logger.info(
            "The server and all clients have been shutted down successfully."
        )

    def cancel_all_tasks(self):
        """Cancel all on-the-fly client tasks."""
        for task_fut in self.executing_task_futs:
            ray.cancel(task_fut)
            task_id = self.executing_task_futs[task_fut]
            client_id = self.executing_tasks[task_id].client_id
            self.logger.info(f"cancelled task id: {task_id}, client id: {client_id}")
        self.executing_task_futs = {}
        self.executing_tasks = {}

    def __update_executing_task(
        self, client_metadata_local, task_id, client_id, task_fut
    ):
        try:
            self.executing_tasks[task_id].end_time = time.time()
            self.executing_tasks[task_id].success = True
            self.executing_tasks[task_id].log = client_metadata_local.get("log", {})
            # Clean up the task
            self.logger.info(
                f"Received results of task '{self.executing_tasks[task_id].task_name}' from {client_id}."
            )
            self.executing_tasks.pop(task_id)
            self.executing_task_futs.pop(task_fut)
        except Exception as e:
            self.logger.error(
                f"Task {self.executing_tasks[task_id].task_name} on {client_id} failed with an error."
            )
            raise e

    def __send_task(self, client: RayClientCommunicator, task_name, model, metadata):
        ref = None
        if task_name == "get_sample_size":
            ref = client.get_sample_size.remote()
        elif task_name == "data_readiness_report":
            ref = client.data_readiness_report.remote()
        elif task_name == "train":
            ref = client.train.remote(model, metadata)
        return str(uuid.uuid4()), ref

    def _default_logger(self):
        """Create a default logger for the gRPC server if no logger provided."""
        return super()._default_logger()

    def _register_task(self, task_id, task_fut, client_id, task_name):
        """
        Register new client task to the list of executing tasks - call after task submission.
        """
        super()._register_task(task_id, task_fut, client_id, task_name)

    def _check_and_initialize_s3(self, server_agent_config):
        super()._check_and_initialize_s3(server_agent_config)

    def _load_proxystore(self, server_agent_config) -> None:
        """
        Create the proxystore for storing and sending model parameters from the server to the clients.
        """
        super()._load_proxystore(server_agent_config)

    def _parse_result(self, result):
        """
        Parse the returned results from a Globus Compute endpoint.
        The results can be composed of two parts:
        - Model parameters (can be model, gradients, compressed model, etc.)
        - Metadata (may contain additional information such as logs, etc.)
        :param `result`: The result returned from the Globus Compute endpoint.
        :return `model`: The model parameters returned from the client
        :return `metadata`: The metadata returned from the client
        """
        return super()._parse_result(result)
