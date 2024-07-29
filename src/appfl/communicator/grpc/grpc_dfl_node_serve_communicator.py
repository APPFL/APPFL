import json
import logging
from typing import Optional
from .grpc_communicator_pb2 import *
from .grpc_communicator_pb2_grpc import *
from appfl.agent import DFLNodeAgent
from appfl.logger import ServerAgentFileLogger
from .utils import proto_to_databuffer, serialize_model

class GRPCDFLNodeServeCommunicator(GRPCCommunicatorServicer):
    """
    `GRPCDFLNodeServeCommunicator`
    This class is used to serve the DFL node using gRPC, which handles
    the requests from its neighbor clients regardings its local model parameters.
    """
    def __init__(
        self,
        dfl_node_agent: DFLNodeAgent,
        max_message_size: int = 2 * 1024 * 1024,
        logger: Optional[ServerAgentFileLogger] = None,
    ) -> None:
        self.server_agent = dfl_node_agent  # this is only used to be consistent with the `serve` function
        self.dfl_node_agent = dfl_node_agent
        self.max_message_size = max_message_size
        self.logger = logger if logger is not None else self._default_logger()

    def GetGlobalModel(self, request, context):
        self.logger.info(
            f"Received GetGlobalModel request from client {request.header.client_id}"
        )
        meta_data = self._extract_metadata_from_request(request)
        model = self.dfl_node_agent.get_parameters(blocking=True, **meta_data)
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

    def InvokeCustomAction(self, request, context):
        self.logger.info(
            f"Received InvokeCustomAction {request.action} request from client {request.header.client_id}"
        )
        client_id = request.header.client_id
        action = request.action
        meta_data = self._extract_metadata_from_request(request)
        if action == "close_connection":
            self.dfl_node_agent.close_connection(client_id)
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