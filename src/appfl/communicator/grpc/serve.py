"""
Serve a gRPC server
"""
import grpc
import logging
from concurrent import futures
from appfl.login_manager import *
from .grpc_communicator_pb2 import *
from .grpc_communicator_pb2_grpc import *
from .utils import load_credential_from_file
from .auth import APPFLAuthMetadataInterceptor
from typing import Any, Optional, Union, Dict

def serve(
    servicer: Any,
    *,
    server_uri: str,
    use_ssl: bool = False,
    use_authenticator: bool = False,
    server_certificate_key: Optional[Union[bytes, str]] = None,
    server_certificate: Optional[Union[bytes, str]] = None,
    authenticator: Optional[str] = None,
    authenticator_args: Dict[str, Any] = {},
    max_message_size: int = 2 * 1024 * 1024,
    max_workers: int = 10,
    **kwargs,
):
    """
    Serve a gRPC servicer.
    :param: server_uri: The uri to serve the gRPC server at.
    :param servicer: The gRPC servicer to serve.
    :param use_ssl: Whether to use SSL/TLS to authenticate the server and encrypt communicated data.
    :param use_authenticator: Whether to use an authenticator to authenticate the client in each RPC. Must have `use_ssl=True` if `True`.
    :param server_certificate_key: The PEM-encoded server certificate key as a byte string, or `None` to use an insecure server.
    :param server_certificate: The PEM-encoded server certificate as a byte string, or `None` to use an insecure server.
    :param authenticator: The name of the authenticator to use for authenticating the client in each RPC.
    :param authenticator_args: The arguments to pass to the authenticator.
    :param max_message_size: The maximum message size in bytes.
    :param max_workers: The maximum number of workers to use for the server.
    """
    assert not (use_authenticator and not use_ssl), "Authenticator can only be used with SSL/TLS"
    if use_ssl:
        assert server_certificate_key is not None, "Server certificate key must be provided if use_ssl is True"
        assert server_certificate is not None, "Server certificate must be provided if use_ssl is True"
    if use_authenticator:
        assert use_ssl, "Authenticator can only be used with SSL/TLS"
        assert authenticator is not None, "Authenticator must be provided if use_authenticator is True"
        authenticator = eval(authenticator)(**authenticator_args)
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ("grpc.max_send_message_length", max_message_size),
            ("grpc.max_receive_message_length", max_message_size),
        ],
        interceptors=(APPFLAuthMetadataInterceptor(authenticator),) if use_authenticator else None,
    )
    add_GRPCCommunicatorServicer_to_server(servicer, server)
    if use_ssl:
        if isinstance(server_certificate_key, str):
            server_certificate_key = load_credential_from_file(server_certificate_key)
        if isinstance(server_certificate, str):
            server_certificate = load_credential_from_file(server_certificate)
        credentials = grpc.ssl_server_credentials(
            (
                (
                    server_certificate_key,
                    server_certificate,
                ),
            )
        )
        server.add_secure_port(server_uri, credentials)
    else:
        server.add_insecure_port(server_uri)
    server.start()
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger = logging.getLogger(__name__)
        logger.info("Terminating the server ...")
        return