from ..grpc_legacy import (
    APPFLgRPCClient,
    APPFLgRPCServer,
    GRPCCommunicator,
    Job,
    grpc_serve,
)
from . import _credentials as _credentials_module
from ._credentials import load_credential_from_file
from .channel import create_grpc_channel
from .grpc_client_communicator import GRPCClientCommunicator
from .grpc_server_communicator import GRPCServerCommunicator
from .serve import serve
from .setup_ssl import setup_ssl
from .utils import deserialize_model, proto_to_databuffer, serialize_model

__all__ = [
    "APPFLgRPCClient",
    "APPFLgRPCServer",
    "GRPCClientCommunicator",
    "GRPCCommunicator",
    "GRPCServerCommunicator",
    "Job",
    "create_grpc_channel",
    "deserialize_model",
    "grpc_serve",
    "load_credential_from_file",
    "proto_to_databuffer",
    "serialize_model",
    "serve",
    "setup_ssl",
]


def __getattr__(name):
    if name in _credentials_module._REMOVED_CERT_NAMES:
        return getattr(_credentials_module, name)
    raise AttributeError(name)
