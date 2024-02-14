import grpc
import json
import pickle
import logging
from typing import Optional
from omegaconf import OmegaConf
from .grpc_communicator_new_pb2 import *
from .grpc_communicator_new_pb2_grpc import *
from appfl.agent import APPFLServerAgent
from appfl.logger import ServerAgentFileLogger
from concurrent import futures
from .utils import proto_to_databuffer_new

class NewGRPCCommunicator(NewGRPCCommunicatorServicer):
    def __init__(
        self,
        server_agent: APPFLServerAgent,
        logger: Optional[ServerAgentFileLogger] = None,
    ) -> None:
        self.server_agent = server_agent
        self.logger = logger if logger is not None else self._default_logger()

    def GetConfiguration(self, request, context):
        self.logger.info(f"Received GetConfiguration request: {request}")
        client_configs = self.server_agent.get_client_configs()
        client_configs = OmegaConf.to_container(client_configs, resolve=True)
        client_configs_serialized = json.dumps(client_configs)
        self.logger.info(f"Sending client configurations: {client_configs_serialized}")
        return ConfigurationResponse(
            header=ServerHeader(
                status=ServerStatus.RUN
            ),
            configuration=client_configs_serialized,
        )
    
    def GetGlobalModel(self, request, context):
        """
        Return the global model to clients
        """
        model = self.server_agent.get_parameters()
        model_serialized = pickle.dumps(model)
        proto = GlobalModelRespone(
            header=ServerHeader(status=ServerStatus.RUN),
            global_model=model_serialized,
        )
        self.logger.info("LLL")
        for bytes in proto_to_databuffer_new(proto):
            yield bytes

    
    def SendLocalModel(self, request_iterator, context):   
        self.logger.info(f"Received SendLocalModel request: {request_iterator[0]}") 
    
    def CustomAction(self, request, context):
        self.logger.info(f"Received CustomAction request: {request}")
    
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
