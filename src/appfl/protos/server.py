from concurrent import futures
import logging

import grpc
from .federated_learning_pb2 import MessageStatus
from .federated_learning_pb2 import Job
from .federated_learning_pb2 import JobResponse
from .federated_learning_pb2 import WeightResponse
from .federated_learning_pb2 import LearningResults
from .federated_learning_pb2 import Acknowledgment
from . import utils
from . import federated_learning_pb2_grpc


class FLServicer(federated_learning_pb2_grpc.FederatedLearningServicer):
    def __init__(self, servicer_id, port, operator):
        self.servicer_id = servicer_id
        self.port = port
        self.operator = operator
        self.logger = logging.getLogger(__name__)

    def GetJob(self, request, context):
        self.logger.debug(
            f"[Servicer ID: {self.servicer_id: 03}] Received JobRequest from client %d job_done %d",
            request.header.client_id,
            request.job_done,
        )
        round_number, job_todo = self.operator.get_job()
        return JobResponse(
            header=request.header, round_number=round_number, job_todo=job_todo
        )

    def GetTensorRecord(self, request, context):
        self.logger.debug(
            f"[Servicer ID: {self.servicer_id: 03}] Received TensorRequest from (client,name,round)=(%d,%s,%d)",
            request.header.client_id,
            request.name,
            request.round_number,
        )
        nparray = self.operator.get_tensor(request.name)
        return utils.construct_tensor_record(request.name, nparray)

    def GetWeight(self, request, context):
        self.logger.debug(
            f"[Servicer ID: {self.servicer_id: 03}] Received WeightRequest from (client,size)=(%d,%d)",
            request.header.client_id,
            request.size,
        )
        weight = self.operator.get_weight(request.header.client_id, request.size)
        self.logger.debug(
            f"[Servicer ID: {self.servicer_id: 03}] get_weight returns %e", weight
        )
        return WeightResponse(header=request.header, weight=weight)

    def SendLearningResults(self, request_iterator, context):
        self.logger.debug(
            f"[Servicer ID: {self.servicer_id: 03}] self.operator.fed_server.weights: {self.operator.fed_server.weights}"
        )
        # Restore LearningResults protocol buffer.
        proto = LearningResults()
        bytes_received = b""
        for request in request_iterator:
            bytes_received += request.data_bytes

        self.logger.debug(
            f"[Servicer ID: {self.servicer_id: 03}] self.operator.fed_server.weights: {self.operator.fed_server.weights}"
        )

        status = MessageStatus.EMPTY
        if len(bytes_received) > 0:
            status = MessageStatus.OK
            proto.ParseFromString(bytes_received)
            self.operator.send_learning_results(
                proto.header.client_id,
                proto.round_number,
                proto.penalty,
                proto.primal,
                proto.dual,
            )

        ack = Acknowledgment(header=proto.header, status=status)
        return ack


def serve(servicer, max_message_size=2 * 1024 * 1024):
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", max_message_size),
            ("grpc.max_receive_message_length", max_message_size),
        ],
    )
    federated_learning_pb2_grpc.add_FederatedLearningServicer_to_server(
        servicer, server
    )
    server.add_insecure_port("[::]:" + servicer.port)
    server.start()
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger = logging.getLogger(__name__)
        logger.info("Terminating the server ...")
        return
