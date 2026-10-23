import gc
import io
import copy
import grpc
import time
import uuid
import yaml
import torch
import random
import string
import pprint
import logging
import threading
from typing import Optional
from datetime import datetime
from omegaconf import OmegaConf
from concurrent.futures import Future, ThreadPoolExecutor
from appfl.comm.utils.s3_utils import extract_model_from_s3, send_model_by_pre_signed_s3
from .grpc_communicator_pb2 import (
    UpdateGlobalModelRequest,
    UpdateGlobalModelResponse,
    ConfigurationResponse,
    GetGlobalModelResponse,
    CustomActionRequest,
    CustomActionResponse,
    ServerHeader,
    ServerStatus,
    DataBuffer,
)
from proxystore.store import Store
from proxystore.proxy import extract
from .grpc_communicator_pb2_grpc import GRPCCommunicatorServicer
from appfl.agent import ServerAgent
from appfl.logger import ServerAgentFileLogger
from appfl.misc.utils import deserialize_yaml, get_proxystore_connector
from .utils import (
    proto_to_databuffer,
    response_chunk_size,
    serialize_model,
    deserialize_model,
)
from .compatibility import (
    APPFL_VERSION,
    APPFL_VERSION_HEADER,
    CLIENT_RULES,
    IncompatibleVersionError,
    check_peer_version,
)
from appfl.misc.memory_utils import (
    efficient_bytearray_concatenation,
    optimize_memory_cleanup,
)


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
        # Check for optimize_memory in kwargs (from grpc_configs), default to True
        self.optimize_memory = kwargs.get("optimize_memory", True)
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
        self._num_connected_clients = 0
        self._total_num_clients = self.server_agent.get_num_clients()

        # Polling mode: acknowledge each request immediately with a PENDING ticket, and let the
        # client poll for the result every `poll_interval` seconds. This keeps every RPC short,
        # so gateways with an idle timeout in front of the server do not cut off waiting clients.
        self.use_polling = kwargs.get("use_polling", False)
        self.poll_interval = kwargs.get("poll_interval", 5.0)
        self._pending_results = {}  # ticket -> (Future of (response, model_bytes), response type, chunk size)
        self._pending_lock = threading.Lock()
        self.client_versions = {}  # client ID -> APPFL version, `None` if older than 1.12.0
        self._incompatible_clients = {}  # client ID -> why its requests are rejected
        self._client_versions_lock = threading.Lock()
        self._executor = (
            ThreadPoolExecutor(max_workers=kwargs.get("max_workers", 128))
            if self.use_polling
            else None
        )

        # Streamed aggregation configuration
        self.use_model_chunking = kwargs.get("use_model_chunking", False)
        self.model_chunk_size = kwargs.get("model_chunk_size", 1 * 1024 * 1024 * 1024)
        if self.use_model_chunking:
            # Verify aggregator supports streamed aggregation
            from appfl.algorithm.aggregator import FedAvgAggregator

            supported_aggregators = [FedAvgAggregator]
            aggregator = server_agent.aggregator

            if not any(
                isinstance(aggregator, agg_type) for agg_type in supported_aggregators
            ):
                supported_names = [agg.__name__ for agg in supported_aggregators]
                raise ValueError(
                    f"Streamed aggregation (use_model_chunking=True) is only supported with "
                    f"{', '.join(supported_names)}, but got {type(aggregator).__name__}. "
                    f"Please use a supported aggregator or disable model chunking."
                )

            # Configure chunk size on the aggregator
            aggregator.set_model_chunk_size(self.model_chunk_size)

            self.logger.debug(
                f"Streamed aggregation enabled on server with {type(aggregator).__name__}, "
                f"chunk size: {self.model_chunk_size / (1024**2):.2f} MB"
            )

    def GetConfiguration(self, request, context):
        """
        Client requests the FL configurations that are shared among all clients from the server.
        :param: `request.header.client_id`: A unique client ID
        :param: `request.meta_data`: YAML serialized metadata dictionary (if needed)
        :return `response.header.status`: Server status
        :return `response.configuration`: YAML serialized FL configurations

        The response is a single message, so it is limited by the message sizes on both sides.
        Clients use `GetConfigurationStream` instead, and this is kept for older clients.
        """
        try:
            response, _ = self._get_configuration_result(request, context)
            return response
        except IncompatibleVersionError as e:
            # Reject the client with why, as its requests would fail anyway
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(e))
        except Exception as e:
            logging.error("An error occurred", exc_info=True)
            # Handle the exception in a way that's appropriate for your application
            # For example, you might want to set a gRPC error status
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Server error occurred!")
            raise e

    def GetConfigurationStream(self, request, context):
        """
        Same as `GetConfiguration`, but the response is sent back as a stream of messages, so that
        configurations of any size fit within the message sizes of both sides.
        """
        try:
            response, chunk_size = self._get_configuration_result(request, context)
            yield from self._stream_response(response, chunk_size=chunk_size)
        except IncompatibleVersionError as e:
            # Reject the client with why, as its requests would fail anyway
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(e))
        except Exception as e:
            logging.error("An error occurred", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Server error occurred!")
            raise e

    def _get_configuration_result(self, request, context):
        """
        Get the FL configurations shared among all clients for a configuration request.
        :return: the response protobuf and the chunk size for streaming it to the client
        """
        self._num_connected_clients += 1
        self.logger.info(
            f"Received [{self._num_connected_clients}/{self._total_num_clients}] GetConfiguration request from client {request.header.client_id}"
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
        chunk_size = self._process_client_meta(
            context, request.header.client_id, meta_data
        )
        client_configs = self.server_agent.get_client_configs(**meta_data)
        client_configs["experiment_id"] = self.experiment_id
        client_configs = OmegaConf.to_container(client_configs, resolve=True)
        client_configs_serialized = yaml.dump(client_configs)
        response = ConfigurationResponse(
            header=ServerHeader(status=ServerStatus.RUN),
            configuration=client_configs_serialized,
        )
        return response, chunk_size

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
            chunk_size = self._process_client_meta(
                context, request.header.client_id, meta_data
            )
            if self.use_polling:
                yield from self._defer(
                    GetGlobalModelResponse,
                    chunk_size,
                    self._get_global_model_result,
                    request.header.client_id,
                    meta_data,
                )
            else:
                yield from self._stream_response(
                    *self._get_global_model_result(request.header.client_id, meta_data),
                    chunk_size=chunk_size,
                )
        except IncompatibleVersionError as e:
            # Reject the client with why, as its requests would fail anyway
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(e))
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

            if self.optimize_memory:
                # Optimized protocol: Parse metadata first, then extract raw model bytes
                data_chunks = []
                chunk_count = 0
                total_bytes = 0
                for bytes_chunk in request_iterator:
                    data_chunks.append(bytes_chunk.data_bytes)
                    total_bytes += len(bytes_chunk.data_bytes)
                    chunk_count += 1
                    if chunk_count % 20 == 0:
                        gc.collect()

                # Parse metadata from first chunk(s)
                metadata_size = 0
                for i in range(1, min(len(data_chunks) + 1, 10)):
                    try:
                        metadata_bytes = b"".join(data_chunks[:i])
                        request.ParseFromString(metadata_bytes)
                        metadata_size = len(metadata_bytes)
                        break
                    except Exception:
                        continue

                if metadata_size == 0:
                    raise Exception("Failed to parse request metadata")

                client_id = request.header.client_id
                self.logger.info(
                    f"Received UpdateGlobalModel request from client {client_id}"
                )

                # Check if local model data follows as raw bytes
                if request.local_model == b"":
                    # Extract model bytes from remaining chunks
                    model_buffer = io.BytesIO()
                    bytes_consumed = 0

                    for chunk in data_chunks:
                        if bytes_consumed >= metadata_size:
                            model_buffer.write(chunk)
                        elif bytes_consumed + len(chunk) > metadata_size:
                            offset = metadata_size - bytes_consumed
                            model_buffer.write(chunk[offset:])
                        bytes_consumed += len(chunk)

                    del data_chunks
                    gc.collect()

                    model_buffer.seek(0)
                    model_bytes = model_buffer.read()
                    del model_buffer

                    local_model = model_bytes
                else:
                    local_model = request.local_model
                    del data_chunks
            else:
                # Original protocol: concatenate all and parse
                data_chunks = []
                chunk_count = 0
                for bytes_chunk in request_iterator:
                    data_chunks.append(bytes_chunk.data_bytes)
                    chunk_count += 1

                bytes_received = efficient_bytearray_concatenation(
                    data_chunks, optimize_memory=False
                )
                request.ParseFromString(bytes_received)

                optimize_memory_cleanup(data_chunks, bytes_received, force_gc=True)
                del data_chunks, bytes_received

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
            chunk_size = self._process_client_meta(context, client_id, meta_data)

            # Streamed aggregation: chunk metadata is passed through to aggregator
            if self.use_model_chunking and "_chunk_idx" in meta_data:
                self.logger.info(
                    f"Streamed aggregation: processing chunk [{meta_data['_chunk_idx'] + 1}/"
                    f"{meta_data['_total_chunks']}] from client {client_id}"
                )

            # Memory optimization: Avoid deepcopy when possible
            if len(meta_data) > 0:
                if self.optimize_memory:
                    # Create shallow copy with selective filtering to avoid full deepcopy
                    meta_data_print = {
                        k: v
                        for k, v in meta_data.items()
                        if k
                        not in [
                            "_use_proxystore",
                            "_use_colab_connector",
                            "_use_s3",
                            "model_key",
                            "model_url",
                            "_chunk_idx",
                            "_total_chunks",
                            "_chunk_keys",
                        ]
                    }
                    if (
                        meta_data_print
                    ):  # Only log if there's something meaningful to show
                        # For chunked aggregation, only log for the final chunk
                        if self.use_model_chunking and "_chunk_idx" in meta_data:
                            if (
                                meta_data["_chunk_idx"]
                                == meta_data["_total_chunks"] - 1
                            ):
                                self.logger.info(
                                    f"Received metadata from {request.header.client_id}:\n{pprint.pformat(meta_data_print)}"
                                )
                        else:
                            self.logger.info(
                                f"Received metadata from {request.header.client_id}:\n{pprint.pformat(meta_data_print)}"
                            )
                    del meta_data_print  # Immediate cleanup
                else:
                    # Original behavior with deepcopy
                    meta_data_print = copy.deepcopy(meta_data)
                    remove_keys = [
                        "_use_proxystore",
                        "_use_colab_connector",
                        "_use_s3",
                        "model_key",
                        "model_url",
                        "_chunk_keys",
                        "_chunk_idx",
                        "_total_chunks",
                    ]
                    for key in remove_keys:
                        if key in meta_data_print:
                            del meta_data_print[key]
                    # For chunked aggregation, only log for the final chunk
                    if meta_data_print:
                        if self.use_model_chunking and "_chunk_idx" in meta_data:
                            if (
                                meta_data["_chunk_idx"]
                                == meta_data["_total_chunks"] - 1
                            ):
                                self.logger.info(
                                    f"Received metadata from {request.header.client_id}:\n{pprint.pformat(meta_data_print)}"
                                )
                        else:
                            self.logger.info(
                                f"Received metadata from {request.header.client_id}:\n{pprint.pformat(meta_data_print)}"
                            )

            if self.use_polling:
                yield from self._defer(
                    UpdateGlobalModelResponse,
                    chunk_size,
                    self._update_global_model_result,
                    client_id,
                    local_model,
                    meta_data,
                )
            else:
                response, global_model_serialized = self._update_global_model_result(
                    client_id, local_model, meta_data
                )

                # Memory optimization: Clear local model before streaming the response
                if self.optimize_memory:
                    del local_model
                    gc.collect()

                yield from self._stream_response(
                    response, global_model_serialized, chunk_size=chunk_size
                )
        except IncompatibleVersionError as e:
            # Reject the client with why, as its requests would fail anyway
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(e))
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
            chunk_size = self._process_client_meta(context, client_id, meta_data)
            # `close_connection` returns at once, and the server may shut down right after the
            # last one, before a client could poll for its result, so it is never deferred.
            if self.use_polling and action != "close_connection":
                yield from self._defer(
                    CustomActionResponse,
                    chunk_size,
                    self._custom_action_result,
                    client_id,
                    action,
                    meta_data,
                )
            else:
                yield from self._stream_response(
                    *self._custom_action_result(client_id, action, meta_data),
                    chunk_size=chunk_size,
                )
        except IncompatibleVersionError as e:
            # Reject the client with why, as its requests would fail anyway
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(e))
        except Exception as e:
            logging.error("An error occurred", exc_info=True)
            # Handle the exception in a way that's appropriate for your application
            # For example, you might want to set a gRPC error status
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Server error occurred!")
            raise e

    def PollResult(self, request, context):
        """
        Return the result of a request that the server acknowledged with a PENDING ticket (polling
        mode), or another PENDING response if the result is not ready yet. Returns are sent back as
        a stream of messages, in the same format as the response of the acknowledged request.
        :param: `request.header.client_id`: A unique client ID
        :param: `request.ticket`: The ticket from the PENDING response
        """
        try:
            with self._pending_lock:
                pending = self._pending_results.get(request.ticket)
            if pending is None:
                raise KeyError(
                    f"Unknown ticket {request.ticket} from client {request.header.client_id}"
                )
            future, response_type, chunk_size = pending
            if not future.done():
                yield from proto_to_databuffer(
                    self._pending_response(response_type, request.ticket),
                    max_message_size=chunk_size,
                )
                return
            if future.exception() is not None:
                with self._pending_lock:
                    self._pending_results.pop(request.ticket, None)
                raise future.exception()
            yield from self._stream_response(*future.result(), chunk_size=chunk_size)
            # Only drop the result once fully streamed, so the client can poll again if the transfer fails
            with self._pending_lock:
                self._pending_results.pop(request.ticket, None)
        except Exception as e:
            logging.error("An error occurred", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Server error occurred!")
            raise e

    def _get_global_model_result(self, client_id, meta_data):
        """
        Get the global model for a `GetGlobalModel` request, blocking until it is available.
        :return: the response protobuf and the serialized global model
        """
        use_s3 = meta_data.get("_use_s3", False)
        model_key = None
        model_url = None
        if use_s3:
            model_key = meta_data.get("model_key", None)
            model_url = meta_data.get("model_url", None)

        model = self.server_agent.get_parameters(**meta_data, blocking=True)
        response_meta_data = {}
        if isinstance(model, tuple):
            model, response_meta_data = model

        if use_s3:
            model = send_model_by_pre_signed_s3(
                client_id,
                self.experiment_id,
                "grpc",
                model,
                model_key=model_key,
                model_url=model_url,
                logger=self.logger,
            )

        serialized_meta_data = yaml.dump(response_meta_data)
        if self.use_proxystore:
            model = self.proxystore.proxy(model)

        if self.use_colab_connector:
            model = self.colab_connector.upload(
                model, f"init_global_model{int(time.time())}.pt"
            )

        response = GetGlobalModelResponse(
            header=ServerHeader(status=ServerStatus.RUN),
            meta_data=serialized_meta_data,
        )
        return response, serialize_model(model)

    def _update_global_model_result(self, client_id, local_model, meta_data):
        """
        Update the global model with a client's local model for an `UpdateGlobalModel` request,
        blocking until the updated global model is available.
        :return: the response protobuf and the serialized global model
        """
        # Fetching the local model from the transport may take long, so it is done here to
        # be deferred in polling mode, instead of before the PENDING acknowledgment
        if meta_data.get("_use_proxystore", False):
            local_model_proxy = deserialize_model(local_model)
            local_model = extract(local_model_proxy)

            if self.optimize_memory:
                optimize_memory_cleanup(local_model_proxy, force_gc=True)
        if meta_data.get("_use_colab_connector", False):
            local_model = deserialize_model(local_model)
            local_model = self.colab_connector.load_model(
                local_model["model_drive_path"]
            )
        if meta_data.get("_use_s3", False):
            local_model = extract_model_from_s3(
                client_id,
                self.experiment_id,
                "grpc",
                deserialize_model(local_model),
            )

        global_model = self.server_agent.global_update(
            client_id, local_model, blocking=True, **meta_data
        )

        # Memory optimization: Clear local model immediately after processing
        if self.optimize_memory:
            del local_model
            gc.collect()

        response_meta_data = {}
        if isinstance(global_model, tuple):
            global_model, response_meta_data = global_model

        if self.use_proxystore:
            global_model = self.proxystore.proxy(global_model)
        if self.use_colab_connector:
            global_model = self.colab_connector.upload(
                global_model, f"global_model{int(time.time())}.pt"
            )
        if meta_data.get("_use_s3", False):
            global_model = send_model_by_pre_signed_s3(
                client_id,
                self.experiment_id,
                "grpc",
                global_model,
                model_key=meta_data.get("model_key", None),
                model_url=meta_data.get("model_url", None),
                logger=self.logger,
            )

        status = (
            ServerStatus.DONE
            if self.server_agent.training_finished()
            else ServerStatus.RUN
        )
        response = UpdateGlobalModelResponse(
            header=ServerHeader(status=status),
            meta_data=yaml.dump(response_meta_data),
        )
        if self.optimize_memory:
            global_model_serialized = self._serialize_model_optimized(global_model)
        else:
            global_model_serialized = serialize_model(global_model)
        return response, global_model_serialized

    def _custom_action_result(self, client_id, action, meta_data):
        """
        Perform a custom action for an `InvokeCustomAction` request, blocking until it is done.
        :return: the response protobuf and `None` (no model to send back)
        """
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
        return response, None

    def _process_client_meta(self, context, client_id, meta_data):
        """
        Exchange APPFL versions with the client, and process the transport keys in the `meta_data`
        of a request.

        - The server sends its APPFL version in the `appfl-version` response header. The client's
          version, from its request header, is recorded in `client_versions` and checked against
          `CLIENT_RULES` whenever it changes. Clients older than APPFL 1.12.0 send no version.
        - `max_message_size` is popped, as it is for the communicator, not the server agent: the
          response chunk size is the smaller of the server's `max_message_size` and the client's, so
          that every chunk fits the client's receive limit.

        :return: the chunk size for the responses to the request
        :raises IncompatibleVersionError: if the client's version breaks an "error" rule
        """
        context.send_initial_metadata(((APPFL_VERSION_HEADER, APPFL_VERSION),))
        client_version = dict(context.invocation_metadata()).get(APPFL_VERSION_HEADER)
        with self._client_versions_lock:
            changed = (
                client_id not in self.client_versions
                or self.client_versions[client_id] != client_version
            )
            self.client_versions[client_id] = client_version
            if changed:
                errors = []
                for level, message in check_peer_version(
                    f"client {client_id}", client_version, CLIENT_RULES, self
                ):
                    getattr(self.logger, level)(message)
                    if level == "error":
                        errors.append(message)
                if errors:
                    self._incompatible_clients[client_id] = " ".join(errors)
                else:
                    self._incompatible_clients.pop(client_id, None)
            error = self._incompatible_clients.get(client_id)
        if error is not None:
            raise IncompatibleVersionError(error)
        chunk_size = response_chunk_size(self.max_message_size, meta_data)
        meta_data.pop("max_message_size", None)
        return chunk_size

    def _defer(self, response_type, chunk_size, compute, *args):
        """
        Polling mode: run `compute(*args)` in the background, and immediately acknowledge the
        request with a PENDING response carrying the ticket for the client to poll the result.
        """
        ticket = uuid.uuid4().hex
        with self._pending_lock:
            self._pending_results[ticket] = (
                self._executor.submit(compute, *args),
                response_type,
                chunk_size,
            )
        yield from proto_to_databuffer(
            self._pending_response(response_type, ticket),
            max_message_size=chunk_size,
        )

    def _pending_response(self, response_type, ticket):
        return response_type(
            header=ServerHeader(
                status=ServerStatus.PENDING,
                ticket=ticket,
                poll_interval=self.poll_interval,
            )
        )

    def _stream_response(self, response, model_serialized=None, *, chunk_size):
        """
        Stream a response protobuf, and the serialized model (if any), as `DataBuffer` messages
        of at most `chunk_size` bytes.
        """
        if model_serialized is None:
            yield from proto_to_databuffer(response, max_message_size=chunk_size)
        elif self.optimize_memory:
            # Optimized protocol: Stream metadata-only message (empty `global_model` signals
            # that model data follows separately), then the raw model bytes
            response.global_model = b""
            yield from self._proto_to_databuffer_optimized(
                response, max_message_size=chunk_size
            )
            yield from self._stream_raw_bytes(model_serialized, chunk_size)
        else:
            # Original protocol: Embed model in protobuf (backward compatible)
            response.global_model = model_serialized
            yield from proto_to_databuffer(response, max_message_size=chunk_size)

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
        if self._executor is not None:
            self._executor.shutdown(wait=False)

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

    def _serialize_model_optimized(self, model):
        """
        Memory-optimized model serialization using BytesIO with context manager.
        """
        with io.BytesIO() as buffer:
            torch.save(model, buffer)
            serialized_data = buffer.getvalue()
        # Force garbage collection after serialization
        gc.collect()
        return serialized_data

    def _proto_to_databuffer_optimized(self, proto, max_message_size=(2 * 1024 * 1024)):
        """
        Memory-optimized version of proto_to_databuffer with periodic garbage collection.
        """
        from .grpc_communicator_pb2 import DataBuffer

        max_message_size = int(0.9 * max_message_size)
        data_bytes = proto.SerializeToString()
        data_bytes_size = len(data_bytes)
        message_size = min(max_message_size, data_bytes_size)

        chunk_count = 0
        for i in range(0, data_bytes_size, message_size):
            chunk = data_bytes[i : i + message_size]
            msg = DataBuffer(data_bytes=chunk)
            yield msg

            chunk_count += 1
            # Periodic garbage collection for large responses
            if chunk_count % 20 == 0:
                gc.collect()

        # Final cleanup
        del data_bytes
        gc.collect()

    def _stream_raw_bytes(self, data_bytes, max_message_size=(2 * 1024 * 1024)):
        """
        Stream raw bytes as DataBuffer chunks WITHOUT protobuf wrapping.
        This avoids protobuf size limits and parsing overhead for large models.
        """
        chunk_size = int(0.9 * max_message_size)
        total_size = len(data_bytes)

        chunk_count = 0
        for i in range(0, total_size, chunk_size):
            chunk = data_bytes[i : i + chunk_size]
            yield DataBuffer(data_bytes=chunk)

            chunk_count += 1
            # Periodic garbage collection for large transfers
            if chunk_count % 20 == 0:
                gc.collect()

        gc.collect()
