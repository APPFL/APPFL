import json
from mpi4py import MPI
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Union, Tuple, OrderedDict
from .config import MPITaskRequest, MPITaskResponse, MPIServerStatus, MPITask
from .serializer import request_to_byte, byte_to_response, byte_to_model, model_to_byte

class MPIClientCommunicator:
    def __init__(
        self,
        comm,
        server_rank: int,
    ):
        self.comm = comm
        self.comm_rank = comm.Get_rank()
        self.comm_size = comm.Get_size()
        self.server_rank = server_rank

    def get_configuration(self, **kwargs) -> DictConfig:
        """
        Get the federated learning configurations from the server.
        :param kwargs: additional metadata to be sent to the server
        :return: the federated learning configurations
        """
        meta_data = json.dumps(kwargs)
        request = MPITaskRequest(
            meta_data=meta_data,
        )
        request_bytes = request_to_byte(request)
        tag = self.comm_rank + self.comm_size * MPITask.GET_CONFIGURATION.value
        self.comm.Send(request_bytes, dest=self.server_rank, tag=tag)
        response = self._recv_response()
        if response.status == MPIServerStatus.ERROR.value:
            raise Exception("Server returned an error, stopping the client.")
        configuration = OmegaConf.create(response.meta_data)
        return configuration
    
    def get_global_model(self, **kwargs) -> Union[Union[Dict, OrderedDict], Tuple[Union[Dict, OrderedDict], Dict]]:
        """
        Get the global model from the server.
        :param kwargs: additional metadata to be sent to the server
        :return: the global model with additional metadata (if any)
        """
        meta_data = json.dumps(kwargs)
        request = MPITaskRequest(
            meta_data=meta_data,
        )
        request_bytes = request_to_byte(request)
        tag = self.comm_rank + self.comm_size * MPITask.GET_GLOBAL_MODEL.value
        self.comm.Send(request_bytes, dest=self.server_rank, tag=tag)
        response = self._recv_response()
        if response.status == MPIServerStatus.ERROR.value:
            raise Exception("Server returned an error, stopping the client.")
        model = byte_to_model(response.payload)
        meta_data = json.loads(response.meta_data)
        if len(meta_data) == 0:
            return model
        else:
            return model, meta_data
        
    def update_global_model(self, local_model: Union[Dict, OrderedDict, bytes], **kwargs) -> Tuple[Union[Dict, OrderedDict], Dict]:
        """
        Send local model to FL server for global update, and return the new global model.
        :param local_model: the local model to be sent to the server for gloabl aggregation
        :param kwargs: additional metadata to be sent to the server
        :return: the updated global model with additional metadata. Specifically, `meta_data["status"]` is either "RUNNING" or "DONE".
        """
        meta_data = json.dumps(kwargs)
        request = MPITaskRequest(
            payload=model_to_byte(local_model) if not isinstance(local_model, bytes) else local_model,
            meta_data=meta_data,
        )
        request_bytes = request_to_byte(request)
        tag = self.comm_rank + self.comm_size * MPITask.UPDATE_GLOBAL_MODEL.value
        self.comm.Send(request_bytes, dest=self.server_rank, tag=tag)
        response = self._recv_response()
        if response.status == MPIServerStatus.ERROR.value:
            raise Exception("Server returned an error, stopping the client.")
        model = byte_to_model(response.payload)
        meta_data = json.loads(response.meta_data)
        status = "DONE" if response.status == MPIServerStatus.DONE.value else "RUNNING"
        meta_data["status"] = status
        return model, meta_data

    def invoke_custom_action(self, action: str, **kwargs) -> Dict:
        """
        Invoke a custom action on the server.
        :param action: the action to be invoked
        :param kwargs: additional metadata to be sent to the server
        :return: the response from the server (if any)
        """
        kwargs["action"] = action
        meta_data = json.dumps(kwargs)
        request = MPITaskRequest(
            meta_data=meta_data,
        )
        request_bytes = request_to_byte(request)
        tag = self.comm_rank + self.comm_size * MPITask.INVOKE_CUSTOM_ACTION.value
        self.comm.Send(request_bytes, dest=self.server_rank, tag=tag)
        response = self._recv_response()
        if response.status == MPIServerStatus.ERROR.value:
            raise Exception("Server returned an error, stopping the client.")
        if len(response.meta_data) == 0:
            return {}
        else:
            try:
                return json.loads(response.meta_data)
            except:
                return {}

    def _recv_response(self) -> MPITaskResponse:
        status = MPI.Status()
        self.comm.probe(source=self.server_rank, tag=self.comm_rank, status=status)
        count = status.Get_count(MPI.BYTE)
        response_buffer = bytearray(count)
        self.comm.Recv(response_buffer, source=self.server_rank, tag=self.comm_rank)
        response = byte_to_response(response_buffer)
        return response
        