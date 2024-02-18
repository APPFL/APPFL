import io
import grpc
import json
import torch
from .grpc_communicator_new_pb2 import *
from .grpc_communicator_new_pb2_grpc import *
from omegaconf import OmegaConf, DictConfig
from typing import Union, Dict, OrderedDict, Tuple
from appfl.communicator.grpc import proto_to_databuffer_new, serialize_model

class GRPCClientCommunicator:
    """
    gRPC communicator for federated learning clients.
    """
    def __init__(
        self, 
        *,
        client_id,
        server_uri,
        max_message_size,
    ):
        """
        Create a channel to the server and initialize the gRPC stub.
        TODO: When merging with the main branch, we need to use the channel auxiliar function to create the channel. Now we just create an insecure channel.
        """
        self.client_id = client_id
        self.max_message_size = max_message_size
        channel_options = [
            ("grpc.max_send_message_length", max_message_size),
            ("grpc.max_receive_message_length", max_message_size),
        ]
        channel = grpc.insecure_channel(
            server_uri,
            options=channel_options
        )
        grpc.channel_ready_future(channel).result(timeout=60)
        self.stub = NewGRPCCommunicatorStub(channel)

    def get_configuration(self, **kwargs) -> DictConfig:
        """
        Get the federated learning configurations from the server.
        :param kwargs: additional metadata to be sent to the server
        :return: the federated learning configurations
        """
        meta_data = json.dumps(kwargs)
        request = ConfigurationRequest(
            header=ClientHeader(client_id=self.client_id),
            meta_data=meta_data,
        )
        response = self.stub.GetConfiguration(request)
        if response.header.status == ServerStatus.ERROR:
            raise Exception("Server returned an error, stopping the client.")
        configuration = OmegaConf.create(response.configuration)
        return configuration
        
    def get_global_model(self, **kwargs) -> Union[Union[Dict, OrderedDict], Tuple[Union[Dict, OrderedDict], Dict]]:
        """
        Get the global model from the server.
        :param kwargs: additional metadata to be sent to the server
        :return: the global model with additional metadata (if any)
        """
        meta_data = json.dumps(kwargs)
        request = GetGlobalModelRequest(
            header=ClientHeader(client_id=self.client_id),
            meta_data=meta_data,
        )
        byte_received = b''
        for byte in self.stub.GetGlobalModel(request):
            byte_received += byte.data_bytes
        response = GetGlobalModelRespone()
        response.ParseFromString(byte_received)
        if response.header.status == ServerStatus.ERROR:
            raise Exception("Server returned an error, stopping the client.")
        model = torch.load(io.BytesIO(response.global_model))
        meta_data = json.loads(response.meta_data)
        if len(meta_data) == 0:
            return model
        else:
            return model, meta_data

    def update_global_model(self, local_model: Union[Dict, OrderedDict], **kwargs) -> Union[Union[Dict, OrderedDict], Tuple[Union[Dict, OrderedDict], Dict]]:
        """
        Send local model to FL server for global update, and return the new global model.
        :param local_model: the local model to be sent to the server for gloabl aggregation
        :param kwargs: additional metadata to be sent to the server
        :return: the updated global model with additional metadata (if any)
        """
        meta_data = json.dumps(kwargs)
        request = UpdateGlobalModelRequest(
            header=ClientHeader(client_id=self.client_id),
            local_model=serialize_model(local_model),
            meta_data=meta_data,
        )
        byte_received = b''
        for byte in self.stub.UpdateGlobalModel(proto_to_databuffer_new(request, max_message_size=self.max_message_size)):
            byte_received += byte.data_bytes
        response = UpdateGlobalModelResponse()
        response.ParseFromString(byte_received)
        if response.header.status == ServerStatus.ERROR:
            raise Exception("Server returned an error, stopping the client.")
        model = torch.load(io.BytesIO(response.global_model))
        meta_data = json.loads(response.meta_data)
        if len(meta_data) == 0:
            return model
        else:
            return model, meta_data
        
    def invoke_custom_action(self, action: str, **kwargs) -> Dict:
        """
        Invoke a custom action on the server.
        :param action: the action to be invoked
        :param kwargs: additional metadata to be sent to the server
        :return: the response from the server
        """
        meta_data = json.dumps(kwargs)
        request = CustomActionRequest(
            header=ClientHeader(client_id=self.client_id),
            action=action,
            meta_data=meta_data,
        )
        response = self.stub.InvokeCustomAction(request)
        if response.header.status == ServerStatus.ERROR:
            raise Exception("Server returned an error, stopping the client.")
        return json.loads(response.response)