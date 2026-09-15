"""
Auxiliary function to create a secure/insecure gRPC channel.
"""

import logging
import warnings
from typing import Optional

import grpc
from appfl.comm.grpc.auth import APPFLAuthMetadataProvider
from appfl.comm.grpc.channel import (
    _is_ip_literal,
    _uri_host,
    _validate_server_hostname,
)
from appfl.login_manager import BaseAuthenticator

_logger = logging.getLogger(__name__)


def create_grpc_channel(
    server_uri: str,
    *,
    use_ssl: bool = False,
    use_authenticator: bool = False,
    root_certificates: Optional[bytes] = None,
    authenticator: Optional[BaseAuthenticator] = None,
    max_message_size: int = 2 * 1024 * 1024,
    server_hostname: Optional[str] = None,
    insecure_skip_server_identity_check: bool = False,
    allow_uri_hostname_mismatch: bool = False,
) -> grpc.Channel:
    """
    Create a secure/insecure gRPC channel with the given parameters.

    :param server_uri: The URI of the server to connect to.
    :param use_ssl: Whether to use SSL/TLS to authenticate the server and encrypt communicated data.
    :param use_authenticator: Whether to use an authenticator to authenticate the client in each RPC. Must have `use_ssl=True` if `True`.
    :param root_certificates: The PEM-encoded root certificates as a byte string, or `None` to retrieve them from a default location chosen by gRPC runtime.
    :param authenticator: The authenticator to use for authenticating the client in each RPC.
    :param max_message_size: The maximum message size in bytes.
    :param server_hostname: Pinned server identity. See :func:`appfl.comm.grpc.create_grpc_channel`.
    :param insecure_skip_server_identity_check: Opt-out of SAN verification. Loud warning.
    :param allow_uri_hostname_mismatch: Permit URI host vs ``server_hostname`` mismatch.
    :return: The created gRPC channel.
    """
    assert not (use_authenticator and not use_ssl), (
        "Authenticator can only be used with SSL/TLS"
    )
    channel_options = [
        ("grpc.max_send_message_length", max_message_size),
        ("grpc.max_receive_message_length", max_message_size),
    ]
    if use_ssl:
        if not server_hostname and not insecure_skip_server_identity_check:
            raise ValueError(
                "create_grpc_channel: server_hostname is required when "
                "use_ssl=True. Pin the value that appears as a "
                "SubjectAlternativeName on the server certificate. To "
                "intentionally skip the identity check, set "
                "insecure_skip_server_identity_check=True."
            )
        if server_hostname:
            _validate_server_hostname(server_hostname)
            uri_host = _uri_host(server_uri)
            if (
                uri_host
                and not _is_ip_literal(uri_host)
                and uri_host != server_hostname
                and not allow_uri_hostname_mismatch
            ):
                raise ValueError(
                    f"create_grpc_channel: server_uri host {uri_host!r} does "
                    f"not match server_hostname {server_hostname!r}. Set "
                    "allow_uri_hostname_mismatch=True if intentional."
                )
            channel_options.append(("grpc.ssl_target_name_override", server_hostname))
        else:
            msg = (
                "create_grpc_channel: insecure_skip_server_identity_check=True; "
                "server certificate SAN will NOT be verified. Do not use in "
                "production."
            )
            warnings.warn(msg, stacklevel=2)
            _logger.warning(msg)
        if root_certificates is not None:
            credentials = grpc.ssl_channel_credentials(
                root_certificates=root_certificates
            )
        else:
            credentials = grpc.ssl_channel_credentials()
        if use_authenticator:
            assert authenticator is not None, (
                "Authenticator must be provided if use_authenticator is True"
            )
            call_credentials = grpc.metadata_call_credentials(
                APPFLAuthMetadataProvider(authenticator)
            )
            credentials = grpc.composite_channel_credentials(
                credentials, call_credentials
            )
        channel = grpc.secure_channel(server_uri, credentials, options=channel_options)
    else:
        channel = grpc.insecure_channel(server_uri, options=channel_options)
    return channel
