import grpc
import yaml
from .grpc_communicator_pb2 import (
    ClientHeader,
    ConfigurationRequest,
    GetGlobalModelRequest,
    GetGlobalModelRespone,
    UpdateGlobalModelRequest,
    UpdateGlobalModelResponse,
    CustomActionRequest,
    CustomActionResponse,
    ServerStatus,
)
from .grpc_communicator_pb2_grpc import GRPCCommunicatorStub
from omegaconf import OmegaConf, DictConfig
from typing import Union, Dict, OrderedDict, Tuple, Optional, Any
from appfl.comm.grpc import (
    proto_to_databuffer,
    serialize_model,
    deserialize_model,
    create_grpc_channel,
)
from proxystore.store import Store
from proxystore.proxy import Proxy, extract
from appfl.misc.utils import deserialize_yaml, get_proxystore_connector


class GRPCClientCommunicator:
    """
    gRPC communicator for federated learning clients.
    """

    def __init__(
        self,
        client_id: Union[str, int],
        *,
        server_uri: str,
        use_ssl: bool = False,
        use_authenticator: bool = False,
        root_certificate: Optional[Union[str, bytes]] = None,
        authenticator: Optional[str] = None,
        authenticator_args: Dict[str, Any] = {},
        max_message_size: int = 2 * 1024 * 1024,
        **kwargs,
    ):
        """
        Create a channel to the server and initialize the gRPC client stub.

        :param client_id: A unique client ID.
        :param server_uri: The URI of the server to connect to.
        :param use_ssl: Whether to use SSL/TLS to authenticate the server and encrypt communicated data.
        :param use_authenticator: Whether to use an authenticator to authenticate the client in each RPC. Must have `use_ssl=True` if `True`.
        :param root_certificate: The PEM-encoded root certificates as a byte string, or `None` to retrieve them from a default location chosen by gRPC runtime.
        :param authenticator: The name of the authenticator to use for authenticating the client in each RPC.
        :param authenticator_args: The arguments to pass to the authenticator.
        :param max_message_size: The maximum message size in bytes.
        """
        self.client_id = client_id
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
        grpc.channel_ready_future(channel).result(timeout=3600)
        self.stub = GRPCCommunicatorStub(channel)
        self._use_authenticator = use_authenticator
        self.kwargs = kwargs
        self._load_proxystore()

    def get_configuration(self, **kwargs) -> DictConfig:
        """
        Get the federated learning configurations from the server.
        :param kwargs: additional metadata to be sent to the server
        :return: the federated learning configurations
        """
        if "_client_id" in kwargs:
            client_id = str(kwargs["_client_id"])
            del kwargs["_client_id"]
        else:
            client_id = str(self.client_id)
        meta_data = yaml.dump(kwargs)
        request = ConfigurationRequest(
            header=ClientHeader(client_id=client_id),
            meta_data=meta_data,
        )
        response = self.stub.GetConfiguration(request, timeout=3600)
        if response.header.status == ServerStatus.ERROR:
            raise Exception("Server returned an error, stopping the client.")
        configuration = OmegaConf.create(response.configuration)
        return configuration

    def get_global_model(
        self, **kwargs
    ) -> Union[Union[Dict, OrderedDict], Tuple[Union[Dict, OrderedDict], Dict]]:
        """
        Get the global model from the server.
        :param kwargs: additional metadata to be sent to the server
        :return: the global model with additional metadata (if any)
        """
        if "_client_id" in kwargs:
            client_id = str(kwargs["_client_id"])
            del kwargs["_client_id"]
        else:
            client_id = str(self.client_id)
        meta_data = yaml.dump(kwargs)
        request = GetGlobalModelRequest(
            header=ClientHeader(client_id=client_id),
            meta_data=meta_data,
        )
        byte_received = b""
        for byte in self.stub.GetGlobalModel(request, timeout=3600):
            byte_received += byte.data_bytes
        response = GetGlobalModelRespone()
        response.ParseFromString(byte_received)
        if response.header.status == ServerStatus.ERROR:
            raise Exception("Server returned an error, stopping the client.")
        model = deserialize_model(response.global_model)
        if isinstance(model, Proxy):
            model = extract(model)
        meta_data = deserialize_yaml(
            response.meta_data,
            trusted=self.kwargs.get("trusted", False) or self._use_authenticator,
            warning_message="Loading metadata fails due to untrusted data in the metadata, you can fix this by setting `trusted=True` in `grpc_configs` or use an authenticator.",
        )
        if len(meta_data) == 0:
            return model
        else:
            return model, meta_data

    def update_global_model(
        self, local_model: Union[Dict, OrderedDict, bytes], **kwargs
    ) -> Tuple[Union[Dict, OrderedDict], Dict]:
        """
        Send local model to FL server for global update, and return the new global model.
        :param local_model: the local model to be sent to the server for global aggregation
        :param kwargs: additional metadata to be sent to the server
        :return: the updated global model with additional metadata. Specifically, `meta_data["status"]` is either "RUNNING" or "DONE".
        """
        if self.use_proxystore:
            local_model = self.proxystore.proxy(local_model)
            kwargs["_use_proxystore"] = True
        if "_client_id" in kwargs:
            client_id = str(kwargs["_client_id"])
            del kwargs["_client_id"]
        else:
            client_id = str(self.client_id)
        meta_data = yaml.dump(kwargs)
        request = UpdateGlobalModelRequest(
            header=ClientHeader(client_id=client_id),
            local_model=(
                serialize_model(local_model)
                if (
                    isinstance(local_model, Proxy)
                    or (not isinstance(local_model, bytes))
                )
                else local_model
            ),
            meta_data=meta_data,
        )
        byte_received = b""
        for byte in self.stub.UpdateGlobalModel(
            proto_to_databuffer(request, max_message_size=self.max_message_size),
            timeout=3600,
        ):
            byte_received += byte.data_bytes
        response = UpdateGlobalModelResponse()
        response.ParseFromString(byte_received)
        if response.header.status == ServerStatus.ERROR:
            raise Exception("Server returned an error, stopping the client.")
        model = deserialize_model(response.global_model)
        if isinstance(model, Proxy):
            model = extract(model)
        meta_data = deserialize_yaml(
            response.meta_data,
            trusted=self.kwargs.get("trusted", False) or self._use_authenticator,
            warning_message="Loading metadata fails due to untrusted data in the metadata, you can fix this by setting `trusted=True` in `grpc_configs` or use an authenticator.",
        )
        meta_data["status"] = (
            "DONE" if response.header.status == ServerStatus.DONE else "RUNNING"
        )
        return model, meta_data

    def invoke_custom_action(self, action: str, **kwargs) -> Dict:
        """
        Invoke a custom action on the server.
        :param action: the action to be invoked
        :param kwargs: additional metadata to be sent to the server
        :return: the response from the server
        """
        if "_client_id" in kwargs:
            client_id = str(kwargs["_client_id"])
            del kwargs["_client_id"]
        else:
            client_id = str(self.client_id)
        meta_data = yaml.dump(kwargs)
        request = CustomActionRequest(
            header=ClientHeader(client_id=client_id),
            action=action,
            meta_data=meta_data,
        )
        byte_received = b""
        for byte in self.stub.InvokeCustomAction(
            proto_to_databuffer(request, max_message_size=self.max_message_size),
            timeout=3600,
        ):
            byte_received += byte.data_bytes
        response = CustomActionResponse()
        response.ParseFromString(byte_received)
        if response.header.status == ServerStatus.ERROR:
            raise Exception("Server returned an error, stopping the client.")

        if action == "close_connection":
            # Clean-up proxystore
            if hasattr(self, "proxystore") and self.proxystore is not None:
                try:
                    self.proxystore.close(clear=True)
                except:  # noqa: E722
                    self.proxystore.close()

        if len(response.results) == 0:
            return {}
        else:
            try:
                return deserialize_yaml(
                    response.results,
                    trusted=self.kwargs.get("trusted", False)
                    or self._use_authenticator,
                    warning_message="Loading metadata fails due to untrusted data in the metadata, you can fix this by setting `trusted=True` in `grpc_configs` or use an authenticator.",
                )
            except:  # noqa E722
                return {}

    def _load_proxystore(self):
        """
        Create the proxystore for storing and sending model parameters from the server to the clients.
        """
        self.proxystore = None
        self.use_proxystore = False

        if (
            "proxystore_configs" in self.kwargs
            and "enable_proxystore" in self.kwargs["proxystore_configs"]
            and self.kwargs["proxystore_configs"]["enable_proxystore"]
        ):
            self.use_proxystore = True
            self.proxystore = Store(
                name="server-proxystore",
                connector=get_proxystore_connector(
                    self.kwargs["proxystore_configs"]["connector_type"],
                    self.kwargs["proxystore_configs"]["connector_configs"],
                ),
            )
