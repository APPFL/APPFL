import logging
import numpy as np

import grpc

from .federated_learning_pb2 import Header
from .federated_learning_pb2 import DataBuffer
from .federated_learning_pb2 import JobRequest
from .federated_learning_pb2 import LearningResults
from .federated_learning_pb2 import TensorRequest
from .federated_learning_pb2 import TensorRecord
from .federated_learning_pb2_grpc import FederatedLearningStub
from . import utils

class FLClient():
    def __init__(self, client_id, server_uri):
        self.client_id = client_id
        self.channel = grpc.insecure_channel(server_uri)
        self.stub = FederatedLearningStub(self.channel)
        self.header = Header(server_id=1, client_id=self.client_id)
        self.logger = logging.getLogger(__name__)

    def get_job(self, job_done):
        request = JobRequest(header=self.header, job_done=job_done)
        response = self.stub.GetJob(request)
        self.logger.info("Received JobReponse with (server,round,job)=(%d,%d,%d)",
                         response.header.server_id, response.round_number, response.job_todo)
        return response.round_number, response.job_todo

    def get_tensor_record(self, name, round_number):
        request = TensorRequest(header=self.header, name=name, round_number=round_number)
        response = self.stub.GetTensorRecord(request)
        shape = tuple(response.data_shape)
        flat = np.frombuffer(response.data_bytes, dtype=np.float32)
        nparray = np.reshape(flat, newshape=shape, order='C')
        return nparray

    def send_learning_results(self, tensor_dict, round_number):
        tensors = [utils.construct_tensor_record(k, np.array(v)) for k,v in tensor_dict.items()]
        proto = LearningResults(header=self.header, round_number=round_number, tensors=tensors)

        databuffer = []
        databuffer += utils.proto_to_databuffer(proto)
        self.stub.SendLearningResults(iter(databuffer))


