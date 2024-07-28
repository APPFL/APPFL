import grpc
import json
from .grpc_communicator_pb2 import *
from .grpc_communicator_pb2_grpc import *
from typing import Union, Dict, Any, Tuple, OrderedDict, Optional
from appfl.communicator.grpc import deserialize_model, create_grpc_channel

class GRPCDFLNodeConnectCommunicator:
    """
    `GRPCDFLNodeConnectCommunicator`
    This class is used to connect to another DFL node using gRPC to request
    for its local model parameters.
    """
    def __init__(
        self,
        node_id: Union[str, int],
        *,
        server_uri: str,
        use_ssl: bool = False,
        use_authenticator: bool = False,
        root_certificate: Optional[Union[str, bytes]] = None,
        authenticator: Optional[str] = None,
        authenticator_args: Dict[str, Any] = {},
        max_message_size: int = 2 * 1024 * 1024,
        **kwargs,
    ) -> None:
        self.node_id = node_id
        self.max_message_size = max_message_size
        channel = create_grpc_channel(
            server_uri,
            use_ssl=use_ssl,
            use_authenticator=use_authenticator,
            root_certificate=root_certificate,
            authenticator=authenticator,
            authenticator_args=authenticator_args,
            max_message_size=max_message_size,
        )
        grpc.channel_ready_future(channel).result(timeout=3600) # use a long timeout here for DFL
        self.stub = GRPCCommunicatorStub(channel)
    
    def get_neighbor_model(self, **kwargs) -> Union[Union[Dict, OrderedDict], Tuple[Union[Dict, OrderedDict], Dict]]:
        """
        Get the model from the neighbor.
        :param kwargs: additional metadata to be sent to the server
        :return: the neighbor's model with additional metadata (if any)
        """
        if '_node_id' in kwargs:
            node_id = str(kwargs["_node_id"])
            del kwargs["_node_id"]
        else:
            node_id = str(self.node_id)
        request = GetGlobalModelRequest(
            header=ClientHeader(client_id=node_id),
            meta_data=json.dumps(kwargs),
        )
        byte_received = b''
        for byte in self.stub.GetGlobalModel(request, timeout=3600):
            byte_received += byte.data_bytes
        response = GetGlobalModelRespone()
        response.ParseFromString(byte_received)
        if response.header.status == ServerStatus.ERROR:
            raise Exception("Neighbor node returned an error, stopping the node.")
        model = deserialize_model(response.global_model)
        meta_data = json.loads(response.meta_data)
        if len(meta_data) == 0:
            return model
        else:
            return model, meta_data

    def invoke_custom_action(self, action: str, **kwargs) -> Dict:
        """
        Invoke a custom action on the neighbor.
        :param action: the action to be invoked
        :param kwargs: additional metadata to be sent to the neighbor
        :return: the response from the neighbor
        """
        if '_node_id' in kwargs:
            node_id = str(kwargs["_node_id"])
            del kwargs["_node_id"]
        else:
            node_id = str(self.node_id)
        meta_data = json.dumps(kwargs)
        request = CustomActionRequest(
            header=ClientHeader(client_id=node_id),
            action=action,
            meta_data=meta_data,
        )
        response = self.stub.InvokeCustomAction(request, timeout=3600)
        if response.header.status == ServerStatus.ERROR:
            raise Exception("Neighbor node returned an error, stopping the node.")
        if len(response.results) == 0:
            return {}
        else:
            try:
                return json.loads(response.results)
            except:
                return {}