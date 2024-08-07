import json
import logging
import threading
from typing import Optional
from omegaconf import OmegaConf
from .grpc_communicator_pb2 import *
from .grpc_communicator_pb2_grpc import *
from appfl.agent import HFLNodeAgent
from appfl.logger import ServerAgentFileLogger
from appfl.communicator.grpc import GRPCHFLNodeConnectCommunicator
from .utils import proto_to_databuffer, serialize_model

class GRPCHFLNodeServeCommunicator(GRPCCommunicatorServicer):
    """
    `GRPCHFLNodeServeCommunicator`
    This class is used to serve the HFL node which listens to and handles 
    requests from the HFL leaf clients by interacting with the HFL root 
    server using another `GRPCHFLNodeConnectCommunicator` object.
    
    Similar to the `GRPCServerCommunicator` class in typical FL, this class
    supports four types of remote procedure calls (RPCs):
    - `GetConfiguration`: Get the client configurations that should be shared
    with all clients from the HFL root server.
    - `GetGlobalModel`: Get the current global model from the HFL root server.
    - `UpdateGlobalModel`: Update the global model using the local model sent
    from a client.
    - `InvokeCustomAction`: Invoke a custom action, such as closing the connection.
    
    :param hfl_node_agent: The HFL node agent object.
    :param connect_communicator: The communicator object to connect to the HFL root server.
    :param max_message_size: The maximum message size allowed for the gRPC server.
    :param logger: [Optional] A logger object.
    """
    def __init__(
        self,
        hfl_node_agent: HFLNodeAgent,
        connect_communicator: GRPCHFLNodeConnectCommunicator,
        max_message_size: int = 2 * 1024 * 1024,
        logger: Optional[ServerAgentFileLogger] = None,
    ) -> None:
        self.server_agent = hfl_node_agent
        self.hfl_node_agent = hfl_node_agent
        self.connect_communicator = connect_communicator
        self.max_message_size = max_message_size
        self._num_clients = self.hfl_node_agent.get_num_clients()
        self._curr_num_clients = 0
        self._connection_closed = False
        self._get_model_lock = threading.Lock()
        self._close_connection_lock = threading.Lock()
        self._get_configuration_lock = threading.Lock()
        self._update_global_model_lock = threading.Lock()
        self.logger = logger if logger is not None else self._default_logger()
        
    def GetConfiguration(self, request, context):
        self.logger.info(
            f"Received GetConfiguration request from client {request.header.client_id}"
        )
        meta_data = self._extract_metadata_from_request(request)
        with self._get_configuration_lock:
            client_configs = self.hfl_node_agent.get_client_configs(**meta_data)
            if client_configs is None:
                client_configs = self.connect_communicator.get_configuration(**meta_data)
                self.hfl_node_agent.load_client_configs(client_configs)
        client_configs = OmegaConf.to_container(client_configs, resolve=True)
        response = ConfigurationResponse(
            header=ServerHeader(status=ServerStatus.RUN),
            configuration=json.dumps(client_configs),
        )
        return response
    
    def GetGlobalModel(self, request, context):
        self.logger.info(
            f"Received GetGlobalModel request from client {request.header.client_id}"
        )
        meta_data = self._extract_metadata_from_request(request)
        
        model = self.hfl_node_agent.get_parameters(**meta_data)
        if model is None:
            if meta_data.get("init_model", False):
                with self._get_model_lock:
                    # check init model availability again
                    model = self.hfl_node_agent.get_parameters(**meta_data)
                    if model is None:
                        model = self.connect_communicator.get_global_model(**meta_data)
                        self.hfl_node_agent.load_init_parameters(model)
            else:
                model = self.connect_communicator.get_global_model(**meta_data)
        if isinstance(model, tuple):
            model, meta_data = model
        else:
            meta_data = {}
        response = GetGlobalModelRespone(
            header=ServerHeader(status=ServerStatus.RUN),
            global_model=serialize_model(model),
            meta_data=json.dumps(meta_data),
        )
        for bytes in proto_to_databuffer(response, max_message_size=self.max_message_size):
            yield bytes
            
    def UpdateGlobalModel(self, request_iterator, context):
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
        meta_data = self._extract_metadata_from_request(request)
        aggregated_model = self.hfl_node_agent.global_update(
            client_id=client_id,
            local_model=local_model,
            blocking=True,
            **meta_data,
        )
        if isinstance(aggregated_model, tuple):
            aggregated_model, meta_data = aggregated_model
        # Different processes execute sequentially through this block to ensure
        # only one process connects to the server to update the global model
        # TODO: The current implementation is only for synchronous aggregation
        # We may need to think how to support asynchronous HFL.
        with self._update_global_model_lock:
            self._curr_num_clients += 1
            if self._curr_num_clients == 1:
                self.global_model, self.global_update_meta_data = self.connect_communicator.update_global_model(
                    local_model=aggregated_model,
                    **meta_data,
                )
                self.hfl_node_agent.load_updated_model(self.global_model) # Load the updated model locally
            elif self._curr_num_clients == self._num_clients:
                self._curr_num_clients = 0
        global_model_serialized = serialize_model(self.global_model)
        status = ServerStatus.DONE if self.global_update_meta_data["status"] == "DONE" else ServerStatus.RUN
        response = UpdateGlobalModelResponse(
            header=ServerHeader(status=status),
            global_model=global_model_serialized,
            meta_data=json.dumps(self.global_update_meta_data),
        )
        for bytes in proto_to_databuffer(response, max_message_size=self.max_message_size):
            yield bytes
            
    def InvokeCustomAction(self, request, context):
        self.logger.info(
            f"Received InvokeCustomAction {request.action} request from client {request.header.client_id}"
        )
        client_id = request.header.client_id
        action = request.action
        meta_data = self._extract_metadata_from_request(request)
        if action == "close_connection":
            self.hfl_node_agent.close_connection(client_id)
            with self._close_connection_lock:
                if not self._connection_closed:
                    self.connect_communicator.invoke_custom_action(
                        action="close_connection"
                    )
                    self._connection_closed = True
            response = CustomActionResponse(
                header=ServerHeader(status=ServerStatus.DONE),
            )
            return response
        else:
            raise NotImplementedError(
                f"Custom action {action} with metadata {meta_data} is not implemented."
            )
        
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
    
    def _extract_metadata_from_request(self, request):
        """
        Extract metadata from the request.
        """
        if len(request.meta_data) == 0: 
            meta_data = {}
        else:
            meta_data = json.loads(request.meta_data)
        return meta_data