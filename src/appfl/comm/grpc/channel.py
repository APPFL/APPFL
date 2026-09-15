"""
Auxiliary function to create a secure/insecure gRPC channel.
"""

import ipaddress
import logging
import re
import warnings
from typing import Optional, Union, Dict, Any
from urllib.parse import urlsplit

import grpc
from .auth import APPFLAuthMetadataProvider
from .utils import load_credential_from_file
from appfl.misc.utils import get_appfl_authenticator

_logger = logging.getLogger(__name__)

# RFC 1123-ish hostname: dot-separated labels, each label 1-63 chars, total
# <=253 chars, no leading/trailing dash. Underscores rejected.
_RE_HOSTNAME = re.compile(
    r"^(?=.{1,253}$)"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?)"
    r"(?:\.(?:[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?))*$"
)


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _uri_host(server_uri: str) -> Optional[str]:
    """Return the host part of ``server_uri`` (``host:port`` or scheme URL),
    or ``None`` if it can't be parsed."""
    if not server_uri:
        return None
    target = server_uri
    if "://" not in target:
        target = "//" + target
    try:
        parts = urlsplit(target)
    except ValueError:
        return None
    return parts.hostname or None


def _validate_server_hostname(value: str) -> None:
    """Reject obviously bad ``server_hostname`` values at startup.

    Accept: DNS hostname (RFC-1123-ish), IPv4 literal, IPv6 literal.
    Reject: empty, wildcard (``*.foo``), shell metacharacters, whitespace.
    """
    if not value:
        raise ValueError("server_hostname must be non-empty")
    if value.startswith("*"):
        raise ValueError(
            f"server_hostname {value!r} is a wildcard; pin a concrete identity"
        )
    if _is_ip_literal(value):
        return
    if not _RE_HOSTNAME.match(value):
        raise ValueError(
            f"server_hostname {value!r} is not a valid DNS hostname or IP literal"
        )


def create_grpc_channel(
    server_uri: str,
    *,
    use_ssl: bool = False,
    use_authenticator: bool = False,
    root_certificate: Optional[Union[str, bytes]] = None,
    authenticator: Optional[str] = None,
    authenticator_args: Dict[str, Any] = {},
    max_message_size: int = 2 * 1024 * 1024,
    server_hostname: Optional[str] = None,
    insecure_skip_server_identity_check: bool = False,
    allow_uri_hostname_mismatch: bool = False,
    **kwargs,
) -> grpc.Channel:
    """
    Create a secure/insecure gRPC channel with the given parameters.

    :param server_uri: The URI of the server to connect to.
    :param use_ssl: Whether to use SSL/TLS to authenticate the server and encrypt communicated data.
    :param use_authenticator: Whether to use an authenticator to authenticate the client in each RPC. Must have `use_ssl=True` if `True`.
    :param root_certificate: The PEM-encoded root certificates as a byte string, or `None` to retrieve them from a default location chosen by gRPC runtime.
    :param authenticator: The name of the authenticator to use for authenticating the client in each RPC.
    :param authenticator_args: The arguments to pass to the authenticator.
    :param max_message_size: The maximum message size in bytes.
    :param server_hostname: The expected server identity. Required when ``use_ssl=True``
        unless ``insecure_skip_server_identity_check=True`` is also set. The value
        must appear as a DNS or IP SubjectAlternativeName entry on the server cert.
    :param insecure_skip_server_identity_check: If ``True``, do not pin the server
        identity. The chain is still verified against ``root_certificate``, but any
        cert chaining to that CA will be accepted regardless of SAN. Loud warning is
        emitted; never set in production.
    :param allow_uri_hostname_mismatch: If ``True``, accept a ``server_hostname``
        that differs from the DNS authority of ``server_uri``. Defaults to ``False``
        to catch the common "URI points at a load balancer, cert is for the
        backend" misconfiguration at startup instead of attack time.
    :return: The created gRPC channel.
    """
    assert not (use_authenticator and not use_ssl), (
        "Authenticator can only be used with SSL/TLS"
    )
    channel_options = [
        ("grpc.max_send_message_length", max_message_size),
        ("grpc.max_receive_message_length", max_message_size),
        ("grpc.keepalive_time_ms", 30000),
        ("grpc.keepalive_timeout_ms", 20000),
        ("grpc.keepalive_permit_without_calls", 1),
        ("grpc.http2.max_pings_without_data", 0),
        ("grpc.http2.min_time_between_pings_ms", 10000),
    ]
    if use_ssl:
        if not server_hostname and not insecure_skip_server_identity_check:
            raise ValueError(
                "create_grpc_channel: server_hostname is required when "
                "use_ssl=True. Pin the value that appears as a "
                "SubjectAlternativeName on the server certificate (printed "
                "by appfl-setup-ssl). To intentionally skip the identity "
                "check, set insecure_skip_server_identity_check=True."
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
                    f"not match server_hostname {server_hostname!r}. If this "
                    "is intentional (for example connecting through a load "
                    "balancer), set allow_uri_hostname_mismatch=True."
                )
            # `ssl_target_name_override` forces the gRPC C-core to verify the
            # leaf SAN against this value regardless of the URI authority.
            channel_options.append(("grpc.ssl_target_name_override", server_hostname))
        else:
            msg = (
                "create_grpc_channel: insecure_skip_server_identity_check=True; "
                "the server certificate's SAN will NOT be verified. Any cert "
                "chaining to the configured CA will be accepted. Do not use in "
                "production."
            )
            warnings.warn(msg, stacklevel=2)
            _logger.warning(msg)

        if root_certificate is not None:
            if isinstance(root_certificate, str):
                root_certificate = load_credential_from_file(root_certificate)
            credentials = grpc.ssl_channel_credentials(
                root_certificates=root_certificate
            )
        else:
            credentials = grpc.ssl_channel_credentials()
        if use_authenticator:
            assert authenticator is not None, (
                "Authenticator must be provided if use_authenticator is True"
            )
            authenticator = get_appfl_authenticator(authenticator, authenticator_args)
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
