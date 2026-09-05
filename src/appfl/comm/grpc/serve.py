"""
Serve a gRPC server
"""

import ipaddress
import logging
import time
import warnings
import grpc
from concurrent import futures
from .channel import _uri_host
from .grpc_communicator_pb2_grpc import add_GRPCCommunicatorServicer_to_server
from .utils import load_credential_from_file
from .auth import APPFLAuthMetadataInterceptor
from typing import Any, Optional, Union, Dict
from appfl.misc.utils import get_appfl_authenticator

_logger = logging.getLogger(__name__)


def _warn_if_bind_uri_not_covered_by_san(
    server_uri: str, server_certificate: Union[bytes, str]
) -> None:
    """Advisory only — log a WARNING if the bind authority of ``server_uri``
    is not covered by any DNS or IP SAN entry on the server cert. Clients
    pinning ``server_hostname`` would fail handshakes in this case."""
    try:
        from cryptography import x509  # transitive via grpcio
    except ImportError:
        return
    try:
        pem = (
            server_certificate
            if isinstance(server_certificate, (bytes, bytearray))
            else server_certificate.encode()
        )
        cert = x509.load_pem_x509_certificate(pem)
        san_ext = cert.extensions.get_extension_for_class(
            x509.SubjectAlternativeName
        ).value
    except Exception:
        return
    uri_host = _uri_host(server_uri)
    if not uri_host:
        return
    dns_names = set(san_ext.get_values_for_type(x509.DNSName))
    ip_values = {str(ip) for ip in san_ext.get_values_for_type(x509.IPAddress)}
    try:
        ip_canonical = str(ipaddress.ip_address(uri_host))
        is_ip = True
    except ValueError:
        ip_canonical = uri_host
        is_ip = False
    covered = (uri_host in dns_names) or (is_ip and ip_canonical in ip_values)
    if not covered:
        msg = (
            f"serve: bind URI host {uri_host!r} is not covered by any SAN on "
            f"the server certificate (DNS={sorted(dns_names)}, "
            f"IP={sorted(ip_values)}). Clients pinning server_hostname to "
            "this value will reject the handshake."
        )
        warnings.warn(msg, stacklevel=2)
        _logger.warning(msg)


def serve(
    servicer: Any,
    *,
    server_uri: str,
    use_ssl: bool = False,
    use_authenticator: bool = False,
    server_certificate_key: Optional[Union[bytes, str]] = None,
    server_certificate: Optional[Union[bytes, str]] = None,
    ca_certificate: Optional[Union[bytes, str]] = None,
    authenticator: Optional[str] = None,
    authenticator_args: Dict[str, Any] = {},
    max_message_size: int = 2 * 1024 * 1024,
    max_workers: int = 128,
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
    :param ca_certificate: The PEM-encoded CA certificate as a byte string, or `None` to use an insecure server.
    :param authenticator: The name of the authenticator to use for authenticating the client in each RPC.
    :param authenticator_args: The arguments to pass to the authenticator.
    :param max_message_size: The maximum message size in bytes.
    :param max_workers: The maximum number of workers to use for the server.
    """
    assert not (use_authenticator and not use_ssl), (
        "Authenticator can only be used with SSL/TLS"
    )
    if use_ssl:
        assert server_certificate_key is not None, (
            "Server certificate key must be provided if use_ssl is True"
        )
        assert server_certificate is not None, (
            "Server certificate must be provided if use_ssl is True"
        )
    if use_authenticator:
        assert use_ssl, "Authenticator can only be used with SSL/TLS"
        assert authenticator is not None, (
            "Authenticator must be provided if use_authenticator is True"
        )
        authenticator = get_appfl_authenticator(authenticator, authenticator_args)
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ("grpc.max_concurrent_streams", max_workers),
            ("grpc.max_send_message_length", max_message_size),
            ("grpc.max_receive_message_length", max_message_size),
            ("grpc.keepalive_time_ms", 60000),
            ("grpc.keepalive_timeout_ms", 20000),
            ("grpc.keepalive_permit_without_calls", 1),
            ("grpc.http2.max_pings_without_data", 0),
            ("grpc.http2.min_ping_interval_without_data_ms", 10000),
        ],
        interceptors=(APPFLAuthMetadataInterceptor(authenticator),)
        if use_authenticator
        else None,
    )
    add_GRPCCommunicatorServicer_to_server(servicer, server)
    if use_ssl:
        if isinstance(server_certificate_key, str):
            server_certificate_key = load_credential_from_file(server_certificate_key)
        if isinstance(server_certificate, str):
            server_certificate = load_credential_from_file(server_certificate)
        if isinstance(ca_certificate, str):
            ca_certificate = load_credential_from_file(ca_certificate)
        credentials = grpc.ssl_server_credentials(
            (
                (
                    server_certificate_key,
                    server_certificate,
                ),
            ),
            root_certificates=ca_certificate,
        )
        _warn_if_bind_uri_not_covered_by_san(server_uri, server_certificate)
        server.add_secure_port(server_uri, credentials)
    else:
        server.add_insecure_port(server_uri)
    server.start()
    try:
        while True:
            time.sleep(1)
            if servicer.server_agent.server_terminated():
                servicer.cleanup()
                if hasattr(servicer.server_agent, "logger"):
                    servicer.server_agent.logger.info("Terminating the server ...")
                else:
                    print("Terminating the server ...")
                time.sleep(
                    10
                )  # sleep for 10 seconds to ensure clients receive the termination signal
                server.stop(0)
                break
    except KeyboardInterrupt:
        servicer.cleanup()
        if hasattr(servicer.server_agent, "logger"):
            servicer.server_agent.logger.info("Terminating the server ...")
        else:
            print("Terminating the server ...")
        return
