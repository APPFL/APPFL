import time
import uuid
import os.path as osp
import concurrent.futures
from collections import OrderedDict
from omegaconf import DictConfig, OmegaConf
from globus_compute_sdk import Executor, Client
from .utils.endpoint import GlobusComputeClientEndpoint
from .utils.s3_storage import CloudStorage, LargeObjectWrapper
from .utils.config import ClientTask

import pathlib
import logging
from concurrent.futures import Future, as_completed
from typing import Optional, Dict, List, Union, Tuple
from appfl.agent import APPFLServerAgent
from appfl.config import ClientAgentConfig
from appfl.logger import ServerAgentFileLogger


class GlobusComputeServerCommunicator:
    """
    Communicator used by the federated learning server which plans to use Globus Compute
    for orchestrating the federated learning experiments.

    Globus Compute is a distributed function-as-a-service platform that allows users to run
    functions on specified remote endpoints. For more details, check the Globus Compute SDK 
    documentation at https://globus-compute.readthedocs.io/en/latest/endpoints.html.

    :param `gcc`: Globus Compute client object
    :param `server_agent`: APPFL server agent object
    :param `client_agent_configs`: A list of client agent configurations.
    :param [Optional] `logger`: Optional logger object
    :param [Optional] `s3_bucket`: Optional S3 bucket name for large model transfer. It should be noted that Globus 
        Compute only supports 5MB parameter transfer. If the model size is larger, it should be uploaded to
        an S3 bucket and downloaded by the clients.
    :param [Optional] `s3_creds_file`: A file containing the credentials to access the S3 bucket
    :param [Optional] `s3_temp_dir`: Temporary directory for S3 bucket operations
    """
    def __init__(
        self,
        server_agent: APPFLServerAgent,
        client_agent_configs: List[ClientAgentConfig],
        logger: Optional[ServerAgentFileLogger] = None,
        *,
        s3_bucket: Optional[str] = None,
        s3_creds_file: Optional[str] = None,
        s3_temp_dir: str = str(pathlib.Path.home() / ".appfl" / "s3_tmp_dir"),
        **kwargs,
    ):
        gcc = Client()
        self.gce = Executor(funcx_client=gcc) # Globus Compute Executor
        self.server_agent = server_agent
        self.logger = logger if logger is not None else self._default_logger()
        client_config_from_server = self.server_agent.get_client_configs()
        """Initiate the Globus Compute client endpoints."""
        self.client_endpoints: Dict[str, GlobusComputeClientEndpoint] = {}
        for client_config in client_agent_configs:
            assert hasattr(client_config, "endpoint_id"), "Client configuration must have an endpoint_id."
            client_endpoint_id = client_config.endpoint_id
            del client_config.endpoint_id
            self.client_endpoints[client_endpoint_id] = GlobusComputeClientEndpoint(
                client_endpoint_id, 
                OmegaConf.merge(client_config_from_server, client_config),
            )
        """Initilize the S3 bucket for large model transfer if necessary."""
        self.use_s3bucket = s3_bucket is not None
        if self.use_s3bucket:
            self.logger.info(f'Using S3 bucket {s3_bucket} for model transfer.')
            CloudStorage.init(s3_bucket, s3_creds_file, s3_temp_dir, self.logger)
        
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
        if self.use_s3bucket:
            model_wrapper = LargeObjectWrapper(
                data=model,
                name=str(uuid.uuid4()) + "_server_state",
            )
            if not model_wrapper.can_send_directly:
                model = CloudStorage.upload_object(model, register_for_clean=True)
        for i, client_endpoint_id in enumerate(self.client_endpoints):
            client_metadata = (
                metadata[i] if isinstance(metadata, list) else metadata
            )
            if need_model_response and self.use_s3bucket:
                local_model_key = f"{str(uuid.uuid4())}_client_state_{client_endpoint_id}"
                local_model_url = CloudStorage.presign_upload_object(local_model_key)
                client_metadata["local_model_key"] = local_model_key
                client_metadata["local_model_url"] = local_model_url
            task_id, task_future = self.client_endpoints[client_endpoint_id].submit_task(
                self.gce,
                task_name,
                model,
                client_metadata,
            )
            self.__register_task(task_id, task_future, client_endpoint_id, task_name)
            self.logger.info(f"Task '{task_name}' is assigned to {client_endpoint_id}.")

    def send_task_to_one_client(
        self,
        client_endpoint_id: str,
        task_name: str,
        *,
        model: Optional[Union[Dict, OrderedDict, bytes]] = None,
        metadata: Optional[Dict] = {},
        need_model_response: bool = False,
    ):
        """
        Send a specific task to one specific client endpoint.
        :param `client_endpoint_id`: The client endpoint id to which the task is sent.
        :param `task_name`: Name of the task to be executed on the clients
        :param [Optional] `model`: Model to be sent to the clients
        :param [Optional] `metadata`: Additional metadata to be sent to the clients
        :param `need_model_response`: Whether the task requires a model response from the clients
            If so, the server will provide a pre-signed URL for the clients to upload the model if using S3.
        """
        if self.use_s3bucket:
            model_wrapper = LargeObjectWrapper(
                data=model,
                name=str(uuid.uuid4()) + "_server_state",
            )
            if not model_wrapper.can_send_directly:
                model = CloudStorage.upload_object(model, register_for_clean=True)
        if need_model_response and self.use_s3bucket:
            local_model_key = f"{str(uuid.uuid4())}_client_state_{client_endpoint_id}"
            local_model_url = CloudStorage.presign_upload_object(local_model_key)
            metadata["local_model_key"] = local_model_key
            metadata["local_model_url"] = local_model_url
        task_id, task_future = self.client_endpoints[client_endpoint_id].submit_task(
            self.gce,
            task_name,
            model,
            metadata,
        )
        self.__register_task(task_id, task_future, client_endpoint_id, task_name)
        self.logger.info(f"Task '{task_name}' is assigned to {client_endpoint_id}.")

    def recv_result_from_all_clients(self) -> Tuple[Dict, Dict]:
        """
        Receive task results from all clients that have running tasks.
        """
        client_results, client_logs = {}, {}
        while len(self.executing_task_futs):
            fut = next(as_completed(list(self.executing_task_futs)))
            task_id = self.executing_task_futs[fut]
            try:
                result = fut.result()
                client_endpoint_id = self.executing_tasks[task_id].client_endpoint_id
                client_res, client_log = self.__parse_globus_compute_result(result, task_id)
                

    def __parse_globus_compute_result(self, result, task_id, do_finalize=True):
        """
        Parse the returned results from Globus Compute endpoint.
        The results can be composed to two parts: the actual results for the
        task and the client logs throughout the task execution for monitoring.
        :param `result`: The result returned from the Globus Compute endpoint.
        :param `task_id`: The ID of the task being executed.
        """
        # Obtain the client logs.
        client_log = {}
        if type(result) == tuple:
            result, client_log = result
        else:
            client_log = None
        # Download client results from S3 bucker if necessary
        res = self.__process_client_results(res)
        # Change the status of the finished task
        self.__set_task_success_status(task_id, time.time(), client_log)
        # Log the receipt of the task results
        client_name = self.clients[self.executing_tasks[task_id].client_idx].client_cfg.name
        client_task_name = self.executing_tasks[task_id].task_name
        self.logger.info("Recieved results of task '%s' from %s." %(client_task_name, client_name))
        # Finalize the experiment if necessary
        if do_finalize:
            self.__finalize_task(task_id)
        return res, client_log

    def __process_client_results(self, results):
        """Process results from clients, download file from S3 if necessary"""
        if not self.use_s3bucket:
            return results
        for key in results:
            if CloudStorage.is_cloud_storage_object(results[key]):
                results[key] = CloudStorage.download_object(results[key], to_device=self.cfg.server.device, delete_cloud=True, delete_local=True)
        return results

    def receive_sync_endpoints_updates(self):
        """Receive synchronous updates from all client endpoints."""
        client_results = []
        client_logs    = OrderedDict()
        while len(self.executing_task_futs):
            fut = next(concurrent.futures.as_completed(list(self.executing_task_futs)))
            task_id = self.executing_task_futs[fut]
            try:
                result = fut.result()
                client_idx = self.executing_tasks[task_id].client_idx
                client_local_result, client_logs[client_idx] = self.__handle_globus_compute_result(result, task_id)
                client_results.append(client_local_result)
                del self.executing_task_futs[fut]
            except Exception as exc:
                self.logger.error("Task %s on %s is failed with an error." % (self.executing_tasks[task_id].task_name, self.cfg.clients[self.executing_tasks[task_id].client_idx].name))
                raise exc
        return client_results, client_logs

    def __register_task(self, task_id, task_fut, client_endpoint_id, task_name):
        """
        Register new client task to the list of executing tasks - call after task submission.
        """
        self.executing_tasks[task_id] = OmegaConf.structured(
            ClientTask(
                task_id = task_id,
                task_name = task_name,
                client_endpoint_id = client_endpoint_id,
                start_time = time.time()
            )
        )
        self.executing_task_futs[task_fut] = task_id


    def _default_logger(self):
        """Create a default logger for the gRPC server if no logger provided."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter('[%(asctime)s %(levelname)-4s server]: %(message)s')
        s_handler = logging.StreamHandler()
        s_handler.setLevel(logging.INFO)
        s_handler.setFormatter(fmt)
        logger.addHandler(s_handler)
        return logger