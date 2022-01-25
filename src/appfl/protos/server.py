from concurrent import futures
import logging

import grpc
from .federated_learning_pb2 import MessageStatus
from .federated_learning_pb2 import Job
from .federated_learning_pb2 import JobResponse
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
        self.logger.info("Received JobRequest from client %d job_done %d",
                         request.header.client_id, request.job_done)
        round_number, job_todo = self.operator.get_job()
        return JobResponse(header=request.header, round_number=round_number, job_todo=job_todo)

    def GetTensorRecord(self, request, context):
        self.logger.info("Received TensorRequest from (client,name,round)=(%d,%s,%d)",
                         request.header.client_id, request.name, request.round_number)
        nparray = self.operator.get_tensor(request.name)
        return utils.construct_tensor_record(request.name, nparray)

    def SendLearningResults(self, request_iterator, context):
        # Restore LearningResults protocol buffer.
        proto = LearningResults()
        bytes_received = b''
        for request in request_iterator:
            bytes_received += request.data_bytes

        status = MessageStatus.EMPTY
        if len(bytes_received) > 0:
            status = MessageStatus.OK
            proto.ParseFromString(bytes_received)
            self.operator.send_learning_results(proto.header.client_id, proto.round_number, proto.tensors)

        ack = Acknowledgment(header=proto.header, status=status)
        return ack

def serve(servicer):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    federated_learning_pb2_grpc.add_FederatedLearningServicer_to_server(servicer, server)
    server.add_insecure_port('[::]:' + servicer.port)
    server.start()
    server.wait_for_termination()
