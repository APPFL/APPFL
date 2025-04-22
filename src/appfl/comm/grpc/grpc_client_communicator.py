import os
import uuid
import grpc
import time
import yaml
import pathlib
from datetime import datetime
from appfl.comm.utils.s3_storage import CloudStorage
from appfl.comm.utils.s3_utils import extract_model_from_s3, send_model_by_s3
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
        logger: Optional[Any] = None,
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
        self.logger = logger
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
        self._load_google_drive()
        self._s3_initalized = False

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
        # initializing s3 here as we need experiment id so that we can keep track of the models
        self._check_and_initialize_s3(
            experiment_id=configuration.get("experiment_id", None)
        )
        return configuration

    def get_global_model(
        self, **kwargs
    ) -> Union[Union[Dict, OrderedDict], Tuple[Union[Dict, OrderedDict], Dict]]:
        """
        Get the global model from the server.
        :param kwargs: additional metadata to be sent to the server
        :return: the global model with additional metadata (if any)
        """
        self._check_and_initialize_s3()
        if "_client_id" in kwargs:
            client_id = str(kwargs["_client_id"])
            del kwargs["_client_id"]
        else:
            client_id = str(self.client_id)
        if self.use_s3bucket:
            local_model_key = (
                f"{self.experiment_id}_{str(uuid.uuid4())}_server_state_{client_id}"
            )
            local_model_url = CloudStorage.presign_upload_object(
                local_model_key, register_for_clean=True
            )
            kwargs["model_key"] = local_model_key
            kwargs["model_url"] = local_model_url
            kwargs["_use_s3"] = True
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
        if isinstance(model, dict) and "model_drive_path" in model.keys():
            model = self.colab_connector.load_model(model["model_drive_path"])
        if self.use_s3bucket:
            model = extract_model_from_s3(client_id, self.experiment_id, "grpc", model)
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
        self._check_and_initialize_s3()
        if self.use_proxystore:
            local_model = self.proxystore.proxy(local_model)
            kwargs["_use_proxystore"] = True
        if self.use_colab_connector:
            local_model = self.colab_connector.upload(
                local_model, f"local_model_epoch{int(time.time())}.pt"
            )
            kwargs["_use_colab_connector"] = True
        if "_client_id" in kwargs:
            client_id = str(kwargs["_client_id"])
            del kwargs["_client_id"]
        else:
            client_id = str(self.client_id)
        if self.use_s3bucket:
            local_model = send_model_by_s3(
                self.experiment_id, "grpc", local_model, client_id
            )
            local_model_key = (
                f"{self.experiment_id}_{str(uuid.uuid4())}_server_state_{client_id}"
            )
            local_model_url = CloudStorage.presign_upload_object(
                local_model_key, register_for_clean=True
            )
            kwargs["model_key"] = local_model_key
            kwargs["model_url"] = local_model_url
            kwargs["_use_s3"] = True
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
        if isinstance(model, dict) and "model_drive_path" in model.keys():
            model = self.colab_connector.load_model(model["model_drive_path"])
        if self.use_s3bucket:
            model = extract_model_from_s3(client_id, self.experiment_id, "grpc", model)
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

            if self.colab_connector is not None:
                self.colab_connector.cleanup()

            if self.use_s3bucket:
                CloudStorage.clean_up()
                if self.logger is not None:
                    self.logger.info("S3 bucket cleaned up.")

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

    def _load_google_drive(self) -> None:
        self.use_colab_connector = False
        self.colab_connector = None
        if (
            "colab_connector_configs" in self.kwargs
            and "enable" in self.kwargs["colab_connector_configs"]
            and self.kwargs["colab_connector_configs"]["enable"]
        ):
            from appfl.comm.utils.colab_connector import GoogleColabConnector

            self.use_colab_connector = True
            self.colab_connector = GoogleColabConnector(
                self.kwargs["colab_connector_configs"].get(
                    "model_path", "/content/drive/MyDrive/APPFL"
                )
            )

    def _check_and_initialize_s3(self, experiment_id=None):
        if self._s3_initalized:
            return
        self.experiment_id = (
            experiment_id
            if experiment_id is not None
            else datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        )
        self._s3_initalized = True
        # check if s3 enable
        self.use_s3bucket = False
        s3_bucket = None
        if (
            "s3_configs" in self.kwargs
            and "enable_s3" in self.kwargs["s3_configs"]
            and self.kwargs["s3_configs"]["enable_s3"]
        ):
            self.use_s3bucket = True
            s3_bucket = self.kwargs["s3_configs"].get("s3_bucket", None)
            self.use_s3bucket = self.use_s3bucket and s3_bucket is not None
        if self.use_s3bucket:
            if self.logger is not None:
                self.logger.info(f"Using S3 bucket {s3_bucket} for model transfer.")
            s3_creds_file = self.kwargs["s3_configs"].get("s3_creds_file", None)
            s3_temp_dir_default = str(
                pathlib.Path.home() / ".appfl" / "grpc" / "server" / self.experiment_id
            )
            s3_temp_dir = self.kwargs["s3_configs"].get(
                "s3_temp_dir", s3_temp_dir_default
            )
            if not os.path.exists(s3_temp_dir):
                pathlib.Path(s3_temp_dir).mkdir(parents=True, exist_ok=True)
            CloudStorage.init(s3_bucket, s3_creds_file, s3_temp_dir, self.logger)
