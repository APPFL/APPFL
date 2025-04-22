import copy
import grpc
import time
import yaml
import random
import string
import pprint
import logging
import threading
from typing import Optional
from datetime import datetime
from omegaconf import OmegaConf
from concurrent.futures import Future
from appfl.comm.utils.s3_utils import extract_model_from_s3, send_model_by_pre_signed_s3
from .grpc_communicator_pb2 import (
    UpdateGlobalModelRequest,
    UpdateGlobalModelResponse,
    ConfigurationResponse,
    GetGlobalModelRespone,
    CustomActionRequest,
    CustomActionResponse,
    ServerHeader,
    ServerStatus,
)
from proxystore.store import Store
from proxystore.proxy import extract
from .grpc_communicator_pb2_grpc import GRPCCommunicatorServicer
from appfl.agent import ServerAgent
from appfl.logger import ServerAgentFileLogger
from appfl.misc.utils import deserialize_yaml, get_proxystore_connector
from .utils import proto_to_databuffer, serialize_model, deserialize_model


class GRPCServerCommunicator(GRPCCommunicatorServicer):
    def __init__(
        self,
        server_agent: ServerAgent,
        logger: Optional[ServerAgentFileLogger] = None,
        max_message_size: int = 2 * 1024 * 1024,
        **kwargs,
    ) -> None:
        self.server_agent = server_agent
        self.max_message_size = max_message_size
        self.logger = logger if logger is not None else self._default_logger()
        self.kwargs = kwargs
        self.experiment_id = (
            "exp-"
            + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            + "-"
            + "".join(random.choices(string.ascii_lowercase + string.digits, k=2))
        )
        self._load_proxystore(server_agent.server_agent_config)
        self._load_google_drive(server_agent.server_agent_config)

    def GetConfiguration(self, request, context):
        """
        Client requests the FL configurations that are shared among all clients from the server.
        :param: `request.header.client_id`: A unique client ID
        :param: `request.meta_data`: YAML serialized metadata dictionary (if needed)
        :return `response.header.status`: Server status
        :return `response.configuration`: YAML serialized FL configurations
        """
        try:
            self.logger.info(
                f"Received GetConfiguration request from client {request.header.client_id}"
            )
            if len(request.meta_data) == 0:
                meta_data = {}
            else:
                meta_data = deserialize_yaml(
                    request.meta_data,
                    trusted=self.kwargs.get("trusted", False)
                    or self.kwargs.get("use_authenticator", False),
                    warning_message="Loading metadata fails due to untrusted data in the metadata, you can fix this by setting `trusted=True` in `grpc_configs` or use an authenticator.",
                )
            client_configs = self.server_agent.get_client_configs(**meta_data)
            client_configs["experiment_id"] = self.experiment_id
            client_configs = OmegaConf.to_container(client_configs, resolve=True)
            client_configs_serialized = yaml.dump(client_configs)
            response = ConfigurationResponse(
                header=ServerHeader(status=ServerStatus.RUN),
                configuration=client_configs_serialized,
            )
            return response
        except Exception as e:
            logging.error("An error occurred", exc_info=True)
            # Handle the exception in a way that's appropriate for your application
            # For example, you might want to set a gRPC error status
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Server error occurred!")
            raise e

    def GetGlobalModel(self, request, context):
        """
        Return the global model to clients. This method is supposed to be called by clients to get the initial and final global model. Returns are sent back as a stream of messages.
        :param: `request.header.client_id`: A unique client ID
        :param: `request.meta_data`: YAML serialized metadata dictionary (if needed)
        :return `response.header.status`: Server status
        :return `response.global_model`: Serialized global model
        """
        try:
            self.logger.info(
                f"Received GetGlobalModel request from client {request.header.client_id}"
            )
            if len(request.meta_data) == 0:
                meta_data = {}
            else:
                meta_data = deserialize_yaml(
                    request.meta_data,
                    trusted=self.kwargs.get("trusted", False)
                    or self.kwargs.get("use_authenticator", False),
                    warning_message="Loading metadata fails due to untrusted data in the metadata, you can fix this by setting `trusted=True` in `grpc_configs` or use an authenticator.",
                )
            use_s3 = meta_data.get("_use_s3", False)
            if use_s3:
                model_key = meta_data.get("model_key", None)
                model_url = meta_data.get("model_url", None)

            model = self.server_agent.get_parameters(**meta_data, blocking=True)
            meta_data = {}
            if isinstance(model, tuple):
                model, meta_data = model

            if use_s3:
                model = send_model_by_pre_signed_s3(
                    request.header.client_id,
                    self.experiment_id,
                    "grpc",
                    model,
                    model_key=model_key,
                    model_url=model_url,
                    logger=self.logger,
                )

            meta_data = yaml.dump(meta_data)
            if self.use_proxystore:
                model = self.proxystore.proxy(model)

            if self.use_colab_connector:
                model = self.colab_connector.upload(
                    model, f"init_global_model{int(time.time())}.pt"
                )

            model_serialized = serialize_model(model)
            response_proto = GetGlobalModelRespone(
                header=ServerHeader(status=ServerStatus.RUN),
                global_model=model_serialized,
                meta_data=meta_data,
            )
            yield from proto_to_databuffer(
                response_proto, max_message_size=self.max_message_size
            )
        except Exception as e:
            logging.error("An error occurred", exc_info=True)
            # Handle the exception in a way that's appropriate for your application
            # For example, you might want to set a gRPC error status
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Server error occurred!")
            raise e

    def UpdateGlobalModel(self, request_iterator, context):
        """
        Update the global model with the local model from a client. This method will return the updated global model to the client as a stream of messages.
        :param: request_iterator: A stream of `DataBuffer` messages - which contains serialized request in `UpdateGlobalModelRequest` type.

        If concatenating all messages in `request_iterator` to get a `request`, then
        :param: request.header.client_id: A unique client ID
        :param: request.local_model: Serialized local model
        :param: request.meta_data: YAML serialized metadata dictionary (if needed)
        """
        try:
            request = UpdateGlobalModelRequest()
            bytes_received = b""
            for bytes in request_iterator:
                bytes_received += bytes.data_bytes
            request.ParseFromString(bytes_received)
            self.logger.info(
                f"Received UpdateGlobalModel request from client {request.header.client_id}"
            )
            client_id = request.header.client_id
            local_model = request.local_model
            if len(request.meta_data) == 0:
                meta_data = {}
            else:
                meta_data = deserialize_yaml(
                    request.meta_data,
                    trusted=self.kwargs.get("trusted", False)
                    or self.kwargs.get("use_authenticator", False),
                    warning_message="Loading metadata fails due to untrusted data in the metadata, you can fix this by setting `trusted=True` in `grpc_configs` or use an authenticator.",
                )
            if meta_data.get("_use_proxystore", False):
                local_model_proxy = deserialize_model(local_model)
                local_model = extract(local_model_proxy)
            if meta_data.get("_use_colab_connector", False):
                local_model = deserialize_model(local_model)
                local_model = self.colab_connector.load_model(
                    local_model["model_drive_path"]
                )
            use_s3 = meta_data.get("_use_s3", False)
            if use_s3:
                model_key = meta_data.get("model_key", None)
                model_url = meta_data.get("model_url", None)
                local_model = extract_model_from_s3(
                    client_id,
                    self.experiment_id,
                    "grpc",
                    deserialize_model(local_model),
                )
            if len(meta_data) > 0:
                meta_data_print = copy.deepcopy(meta_data)
                remove_keys = [
                    "_use_proxystore",
                    "_use_colab_connector",
                    "_use_s3",
                    "model_key",
                    "model_url",
                ]
                for key in remove_keys:
                    if key in meta_data_print:
                        del meta_data_print[key]
                self.logger.info(
                    f"Received the following meta data from {request.header.client_id}:\n{pprint.pformat(meta_data_print)}"
                )
            global_model = self.server_agent.global_update(
                client_id, local_model, blocking=True, **meta_data
            )
            meta_data = {}
            if isinstance(global_model, tuple):
                global_model, meta_data = global_model

            if self.use_proxystore:
                global_model = self.proxystore.proxy(global_model)
            if self.use_colab_connector:
                global_model = self.colab_connector.upload(
                    global_model, f"global_model{int(time.time())}.pt"
                )
            if use_s3:
                global_model = send_model_by_pre_signed_s3(
                    request.header.client_id,
                    self.experiment_id,
                    "grpc",
                    global_model,
                    model_key=model_key,
                    model_url=model_url,
                    logger=self.logger,
                )

            meta_data = yaml.dump(meta_data)
            global_model_serialized = serialize_model(global_model)
            status = (
                ServerStatus.DONE
                if self.server_agent.training_finished()
                else ServerStatus.RUN
            )
            response = UpdateGlobalModelResponse(
                header=ServerHeader(status=status),
                global_model=global_model_serialized,
                meta_data=meta_data,
            )
            for bytes in proto_to_databuffer(
                response, max_message_size=self.max_message_size
            ):
                yield bytes
        except Exception as e:
            logging.error("An error occurred", exc_info=True)
            # Handle the exception in a way that's appropriate for your application
            # For example, you might want to set a gRPC error status
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Server error occurred!")
            raise e

    def InvokeCustomAction(self, request_iterator, context):
        """
        This function is the entry point for any custom action that the server agent can perform. The server agent should implement the custom action and call this function to perform the action.
        :param: request_iterator: A stream of `DataBuffer` messages - which contains serialized request in `CustomActionRequest` type.

        If concatenating all messages in `request_iterator` to get a `request`, then
        :param: `request.header.client_id`: A unique client ID
        :param: `request.action`: A string tag representing the custom action
        :param: `request.meta_data`: YAML serialized metadata dictionary for the custom action (if needed)
        :return `response.header.status`: Server status
        :return `response.meta_data`: YAML serialized metadata dictionary for return values (if needed)
        """
        try:
            request = CustomActionRequest()
            bytes_received = b""
            for bytes in request_iterator:
                bytes_received += bytes.data_bytes
            request.ParseFromString(bytes_received)

            self.logger.info(
                f"Received InvokeCustomAction {request.action} request from client {request.header.client_id}"
            )
            client_id = request.header.client_id
            action = request.action
            if len(request.meta_data) == 0:
                meta_data = {}
            else:
                meta_data = deserialize_yaml(
                    request.meta_data,
                    trusted=self.kwargs.get("trusted", False)
                    or self.kwargs.get("use_authenticator", False),
                    warning_message="Loading metadata fails due to untrusted data in the metadata, you can fix this by setting `trusted=True` in `grpc_configs` or use an authenticator.",
                )
            if action == "set_sample_size":
                assert "sample_size" in meta_data, (
                    "The metadata should contain parameter `sample_size`."
                )
                ret_val = self.server_agent.set_sample_size(client_id, **meta_data)
                if ret_val is None:
                    response = CustomActionResponse(
                        header=ServerHeader(status=ServerStatus.RUN),
                    )
                else:
                    if isinstance(ret_val, Future):
                        ret_val = ret_val.result()
                    results = yaml.dump(ret_val)
                    response = CustomActionResponse(
                        header=ServerHeader(status=ServerStatus.RUN),
                        results=results,
                    )
            elif action == "close_connection":
                self.server_agent.close_connection(client_id)
                response = CustomActionResponse(
                    header=ServerHeader(status=ServerStatus.DONE),
                )
            elif action == "get_data_readiness_report":
                num_clients = self.server_agent.get_num_clients()
                if not hasattr(self, "_dr_metrics_lock"):
                    self._dr_metrics = {}
                    self._dr_metrics_futures = {}
                    self._dr_metrics_lock = threading.Lock()
                with self._dr_metrics_lock:
                    for k, v in meta_data.items():
                        if k not in self._dr_metrics:
                            self._dr_metrics[k] = {}
                        self._dr_metrics[k][client_id] = v
                    _dr_metric_future = Future()
                    self._dr_metrics_futures[client_id] = _dr_metric_future
                    if len(self._dr_metrics_futures) == num_clients:
                        self.server_agent.data_readiness_report(self._dr_metrics)
                        for client_id, future in self._dr_metrics_futures.items():
                            future.set_result(None)
                        self._dr_metrics = {}
                        self._dr_metrics_futures = {}
                # waiting for the data readiness report to be generated for synchronization
                _dr_metric_future.result()
                response = CustomActionResponse(
                    header=ServerHeader(status=ServerStatus.DONE),
                )
            else:
                raise NotImplementedError(f"Custom action {action} is not implemented.")
            for bytes in proto_to_databuffer(
                response, max_message_size=self.max_message_size
            ):
                yield bytes
        except Exception as e:
            logging.error("An error occurred", exc_info=True)
            # Handle the exception in a way that's appropriate for your application
            # For example, you might want to set a gRPC error status
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Server error occurred!")
            raise e

    def cleanup(self):
        """Cleanup the server communicator."""
        # Clean-up proxystore
        if hasattr(self, "proxystore") and self.proxystore is not None:
            try:
                self.proxystore.close(clear=True)
            except:  # noqa: E722
                self.proxystore.close()
        if self.colab_connector is not None:
            self.colab_connector.cleanup()

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

    def _load_google_drive(self, server_agent_config) -> None:
        self.use_colab_connector = False
        self.colab_connector = None
        if (
            hasattr(server_agent_config.server_configs, "comm_configs")
            and hasattr(
                server_agent_config.server_configs.comm_configs,
                "colab_connector_configs",
            )
            and server_agent_config.server_configs.comm_configs.colab_connector_configs.get(
                "enable", False
            )
        ):
            from appfl.comm.utils.colab_connector import GoogleColabConnector

            self.use_colab_connector = True
            self.colab_connector = GoogleColabConnector(
                server_agent_config.server_configs.comm_configs.colab_connector_configs.get(
                    "model_path", "/content/drive/MyDrive/APPFL"
                )
            )
