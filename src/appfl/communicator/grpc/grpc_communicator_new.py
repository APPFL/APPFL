import io
import grpc
import json
import torch
import logging
from typing import Optional
from concurrent import futures
from omegaconf import OmegaConf
from .grpc_communicator_new_pb2 import *
from .grpc_communicator_new_pb2_grpc import *
from appfl.agent import APPFLServerAgent
from appfl.logger import ServerAgentFileLogger
from .utils import proto_to_databuffer_new, serialize_model

class NewGRPCCommunicator(NewGRPCCommunicatorServicer):
    def __init__(
        self,
        server_agent: APPFLServerAgent,
        max_message_size: int = 2 * 1024 * 1024,
        logger: Optional[ServerAgentFileLogger] = None,
    ) -> None:
        self.server_agent = server_agent
        self.max_message_size = max_message_size
        self.logger = logger if logger is not None else self._default_logger()

    def GetConfiguration(self, request, context):
        """
        Client requests the FL configurations that are shared among all clients from the server.
        :param: `request.header.client_id`: A unique client ID
        :param: `request.meta_data`: JSON serialized metadata dictionary (if needed)
        :return `response.header.status`: Server status
        :return `response.configuration`: JSON serialized FL configurations
        """
        self.logger.info(f"Received GetConfiguration request from client {request.header.client_id}")
        if len(request.meta_data) == 0: 
            meta_data = {}
        else:
            meta_data = json.loads(request.meta_data)
        client_configs = self.server_agent.get_client_configs(**meta_data)
        client_configs = OmegaConf.to_container(client_configs, resolve=True)
        client_configs_serialized = json.dumps(client_configs)
        response = ConfigurationResponse(
            header=ServerHeader(status=ServerStatus.RUN),
            configuration=client_configs_serialized,
        )
        return response
    
    def GetGlobalModel(self, request, context):
        """
        Return the global model to clients. This method is supposed to be called by clients to get the initial and final global model. Returns are sent back as a stream of messages.
        :param: `request.header.client_id`: A unique client ID
        :param: `request.meta_data`: JSON serialized metadata dictionary (if needed)
        :return `response.header.status`: Server status
        :return `response.global_model`: Serialized global model
        """
        self.logger.info(f"Received GetGlobalModel request from client {request.header.client_id}")
        if len(request.meta_data) == 0: 
            meta_data = {}
        else:
            meta_data = json.loads(request.meta_data)
        model = self.server_agent.get_parameters(**meta_data)
        if isinstance(model, tuple):
            model = model[0]
            meta_data = json.dumps(model[1])
        else:
            meta_data = json.dumps({})
        model_serialized = serialize_model(model)
        response_proto = GetGlobalModelRespone(
            header=ServerHeader(status=ServerStatus.RUN),
            global_model=model_serialized,
            meta_data=meta_data,
        )
        for bytes in proto_to_databuffer_new(response_proto, max_message_size=self.max_message_size):
            yield bytes

    def UpdateGlobalModel(self, request_iterator, context):
        """
        Update the global model with the local model from a client. This method will return the updated global model to the client as a stream of messages.
        :param: request_iterator: A stream of `DataBufferNew` messages - which contains serialized request in `UpdateGlobalModelRequest` type.

        If concatenating all messages in `request_iterator` to get a `request`, then
        :param: request.header.client_id: A unique client ID
        :param: request.local_model: Serialized local model
        :param: request.meta_data: JSON serialized metadata dictionary (if needed)
        """
        request = UpdateGlobalModelRequest()
        bytes_received = b""
        for bytes in request_iterator:
            bytes_received += bytes.data_bytes
        request.ParseFromString(bytes_received)
        self.logger.info(f"Received UpdateGlobalModel request from client {request.header.client_id}")
        client_id = request.header.client_id
        local_model = torch.load(io.BytesIO(request.local_model))
        if len(request.meta_data) == 0: 
            meta_data = {}
        else:
            meta_data = json.loads(request.meta_data)
        global_model = self.server_agent.global_update(client_id, local_model, blocking=True, **meta_data)
        if isinstance(global_model, tuple):
            meta_data = json.dumps(global_model[1])
            global_model = global_model[0]
        else:
            meta_data = json.dumps({})
        global_model_serialized = serialize_model(global_model)
        response = UpdateGlobalModelResponse(
            header=ServerHeader(status=ServerStatus.RUN),
            global_model=global_model_serialized,
            meta_data=meta_data,
        )
        for bytes in proto_to_databuffer_new(response, max_message_size=self.max_message_size):
            yield bytes
    
    def CustomAction(self, request, context):
        """
        This function is the entry point for any custom action that the server agent can perform. The server agent should implement the custom action and call this function to perform the action.
        :param: `request.header.client_id`: A unique client ID
        :param: `request.action`: A string tag representing the custom action
        :param: `request.meta_data`: JSON serialized metadata dictionary for the custom action (if needed)
        :return `response.header.status`: Server status
        :return `response.meta_data`: JSON serialized metadata dictionary for return values (if needed)
        """
        self.logger.info(f"Received CustomAction {request.action} request from client {request.header.client_id}")
        client_id = request.header.client_id
        action = request.action
        if len(request.meta_data) == 0: 
            meta_data = {}
        else:
            meta_data = json.loads(request.meta_data)
        self.logger.info(f"[DEBUG] - {meta_data} - {action} - {client_id}")
        response = CustomActionResponse(
            header=ServerHeader(status=ServerStatus.RUN),
            meta_data=json.dumps({"TEST": "TEST"}),
        )
        return response
    
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

def serve(servicer, max_message_size=2 * 1024 * 1024):
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", max_message_size),
            ("grpc.max_receive_message_length", max_message_size),
        ],
    )
    add_NewGRPCCommunicatorServicer_to_server(
        servicer, server
    )
    server.add_insecure_port("localhost:50051")
    server.start()
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger = logging.getLogger(__name__)
        logger.info("Terminating the server ...")
        return
