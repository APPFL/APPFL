import json
from mpi4py import MPI
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Union, Tuple, OrderedDict, Optional, List
from .config import MPITaskRequest, MPITaskResponse, MPIServerStatus, MPITask
from .serializer import request_to_byte, byte_to_response, byte_to_model, model_to_byte

class MPIClientCommunicator:
    """
    MPI client communicator for federated learning.
    
    :param comm: the MPI communicator from mpi4py
    :param server_rank: the rank of the server in the MPI communicator
    :param client_ids: [optional] a list of client IDs for a batched clients,
        this is only required when the MPI process represents multiple clients
    """
    def __init__(
        self,
        comm,
        server_rank: int,
        client_ids: Optional[List[Union[str, int]]] = None,
    ):
        self.comm = comm
        self.comm_rank = comm.Get_rank()
        self.comm_size = comm.Get_size()
        self.server_rank = server_rank
        self.client_ids = (
            [self._id_generator(client_id) for client_id in client_ids]
            if client_ids is not None
            else [self.comm_rank]
        )
        self._raw_client_ids = client_ids if client_ids is not None else [self.comm_rank]
        self._client_ids_to_raw = {
            client_id: raw_client_id for client_id, raw_client_id in zip(self.client_ids, client_ids)
        } if client_ids is not None else {self.comm_rank: self.comm_rank}
        self._default_batching = client_ids is not None

    def get_configuration(
        self, 
        **kwargs
    ) -> DictConfig:
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
    
    def get_global_model(
        self, 
        **kwargs
    ) -> Union[Union[Dict, OrderedDict], Tuple[Union[Dict, OrderedDict], Dict]]:
        """
        Get the global model from the server.
        
        :param kwargs: additional metadata to be sent to the server
        :return: the global model with additional metadata (if any)
        """
        kwargs["_client_ids"] = self.client_ids
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
        
    def update_global_model(
        self, 
        local_model: Union[Dict, OrderedDict, bytes], 
        client_id: Optional[Union[str, int]] = None,
        **kwargs
    ) -> Tuple[Union[Dict, OrderedDict], Dict]:
        """
        Send local model(s) to the FL server for global update, and return the new global model.
        
        :param local_model: the local model to be sent to the server for global aggregation
        
            - `local_model` can be a single model if one MPI process has only one client or one MPI process 
                has multiple clients but the user wants to send one model at a time
            - `local_model` can be a dictionary of multiple models as well if one MPI process has multiple clients 
                and the user wants to send all models 
        :param client_id (optional): the client ID for the local model. It is only required when the MPI process has multiple clients
            and the user only wants to send one model at a time.
        :param kwargs (optional): additional metadata to be sent to the server. When sending local models for multiple clients,
            use the client ID as the key and the metadata as the value, e.g.,
        ```
        update_global_model(
            local_model=...,
            kwargs = {
                client_id1: {key1: value1, key2: value2},
                client_id2: {key1: value1, key2: value2},
            }
        )
        ```
        :return model: the updated global model 
        
            - **Note**: the global model is only one model even if multiple local models are sent, which means that 
            the server should have synchronous aggregation. If asynchronous aggregation is needed, the user should
            pass the local models one by one.
        :return meta_data: additional metadata from the server. When updating local models for multiple clients, the response will 
            be a dictionary with the client ID as the key and the response as the value, e.g.,
        ```
        {
            client_id1: {ret1: value1, ret2: value2},
            client_id2: {ret1: value1, ret2: value2},
        }
        """
        if 'kwargs' in kwargs:
            kwargs = kwargs['kwargs']
        for raw_client_id in self._raw_client_ids:
            if raw_client_id in kwargs:
                kwargs[self._id_generator(raw_client_id)] = kwargs.pop(raw_client_id)
            if raw_client_id in local_model:
                local_model[self._id_generator(raw_client_id)] = local_model.pop(raw_client_id)
        kwargs["_client_ids"] = (
            self.client_ids if client_id is None else [self._id_generator(client_id)]
        )
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
        # post-process the results if the client has multiple clients
        status = "DONE" if response.status == MPIServerStatus.DONE.value else "RUNNING"
        if client_id is not None or (not self._default_batching):
            meta_data = meta_data[str(kwargs["_client_ids"][0])]
            meta_data["status"] = status
        else:
            for id in self.client_ids:
                if id in meta_data:
                    meta_data[self._client_ids_to_raw[id]] = meta_data.pop(id)
                    meta_data[self._client_ids_to_raw[id]]["status"] = status
        return model, meta_data

    def invoke_custom_action(
        self, 
        action: str,
        client_id: Optional[Union[str, int]] = None,
        **kwargs
    ) -> Dict:
        """
        Invoke a custom action on the server.
        
        :param action: the action to be invoked
        :param client_id (optional): the client ID for the action. It is only required when the MPI process has multiple clients
            and the action is specific to a client instead of all clients.
        :param kwargs (optional): additional metadata to be sent to the server. When invoking custom action for multiple clients,
            use the client ID as the key and the metadata as the value, e.g.,
        ```
        invoke_custom_action(
            action=...,
            kwargs = {
                client_id1: {key1: value1, key2: value2},
                client_id2: {key1: value1, key2: value2},
            }
        )
        ```
        :return: the response from the server (if any). When invoking custom action for multiple clients, the response will 
            be a dictionary with the client ID as the key and the response as the value, e.g.,
        ```
        {
            client_id1: {ret1: value1, ret2: value2},
            client_id2: {ret1: value1, ret2: value2},
        }
        """
        # Parse the kwargs if the user passes the kwargs as a dictionary
        if 'kwargs' in kwargs:
            kwargs = kwargs['kwargs']
        for raw_client_id in self._raw_client_ids:
            if raw_client_id in kwargs:
                kwargs[self._id_generator(raw_client_id)] = kwargs.pop(raw_client_id)        
        kwargs["action"] = action
        kwargs["_client_ids"] = (
            self.client_ids if client_id is None else [self._id_generator(client_id)]
        )
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
                results = json.loads(response.meta_data)
                # post-process the results if the client has multiple clients
                if client_id is not None or (not self._default_batching):
                    results = results[str(kwargs["_client_ids"][0])]
                else:
                    for id in self.client_ids:
                        if id in results:
                            results[self._client_ids_to_raw[id]] = results.pop(id)
                return results
            except Exception as e:
                print(e)
                return {}

    def _recv_response(self) -> MPITaskResponse:
        status = MPI.Status()
        self.comm.probe(source=self.server_rank, tag=self.comm_rank, status=status)
        count = status.Get_count(MPI.BYTE)
        response_buffer = bytearray(count)
        self.comm.Recv(response_buffer, source=self.server_rank, tag=self.comm_rank)
        response = byte_to_response(response_buffer)
        return response
        
    def _id_generator(self, client_id) -> str:
        """
        Generate a unique client ID.
        """
        return f"{self.comm_rank}_{client_id}"