"""Tests for src/appfl/comm/grpc/channel.py covering security-review #9
(server identity enforcement via SNI / SubjectAlternativeName)."""

from __future__ import annotations

import datetime
import ipaddress
from typing import Tuple

import pytest

import appfl.comm.grpc.channel as channel_module
from appfl.comm.grpc.channel import (
    _validate_server_hostname,
    create_grpc_channel,
)


# ---------------------------------------------------------------------------
# Hostname validator (pure unit tests, no networking, no fixtures needed).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "good",
    [
        "localhost",
        "appfl.example.com",
        "a.b.c.d.example",
        "127.0.0.1",
        "::1",
        "10.0.0.5",
    ],
)
def test_validate_server_hostname_accepts(good):
    _validate_server_hostname(good)  # does not raise


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "*.example.com",
        "-foo.example",
        "foo_bar.example",
        ".foo",
        "foo..bar",
        "foo bar",
        "foo;rm",
        "foo\nDNS=evil",
        "a" * 254,
    ],
)
def test_validate_server_hostname_rejects(bad):
    with pytest.raises(ValueError):
        _validate_server_hostname(bad)


# ---------------------------------------------------------------------------
# create_grpc_channel — argument-validation tests using a stub grpc.
# ---------------------------------------------------------------------------


class _DummyChannel:
    def close(self):
        pass


@pytest.fixture
def fake_grpc(monkeypatch):
    """Replace grpc.secure_channel/grpc.insecure_channel with recorders so the
    tests never open a socket."""
    calls = {"secure": [], "insecure": []}

    def _secure_channel(target, credentials, options=None):
        calls["secure"].append((target, options))
        return _DummyChannel()

    def _insecure_channel(target, options=None):
        calls["insecure"].append((target, options))
        return _DummyChannel()

    def _ssl_channel_credentials(root_certificates=None):
        return ("ssl-creds", root_certificates)

    monkeypatch.setattr(channel_module.grpc, "secure_channel", _secure_channel)
    monkeypatch.setattr(channel_module.grpc, "insecure_channel", _insecure_channel)
    monkeypatch.setattr(
        channel_module.grpc,
        "ssl_channel_credentials",
        _ssl_channel_credentials,
    )
    return calls


def test_missing_server_hostname_under_tls_raises(fake_grpc):
    with pytest.raises(ValueError, match="server_hostname is required"):
        create_grpc_channel(
            "localhost:50051",
            use_ssl=True,
            root_certificate=None,
        )
    # No socket ever opened.
    assert fake_grpc["secure"] == []
    assert fake_grpc["insecure"] == []


def test_tls_pins_server_hostname_via_override(fake_grpc):
    create_grpc_channel(
        "localhost:50051",
        use_ssl=True,
        server_hostname="localhost",
    )
    assert len(fake_grpc["secure"]) == 1
    _, options = fake_grpc["secure"][0]
    assert ("grpc.ssl_target_name_override", "localhost") in options


def test_ip_literal_uri_with_dns_hostname_is_allowed(fake_grpc):
    """A common HPC case: connect to 127.0.0.1, pin SAN=localhost."""
    create_grpc_channel(
        "127.0.0.1:50051",
        use_ssl=True,
        server_hostname="localhost",
    )
    assert len(fake_grpc["secure"]) == 1


def test_uri_hostname_mismatch_rejected(fake_grpc):
    with pytest.raises(ValueError, match="does not match"):
        create_grpc_channel(
            "internal-lb.corp:443",
            use_ssl=True,
            server_hostname="appfl.example.com",
        )


def test_uri_hostname_mismatch_allowed_with_opt_in(fake_grpc):
    create_grpc_channel(
        "internal-lb.corp:443",
        use_ssl=True,
        server_hostname="appfl.example.com",
        allow_uri_hostname_mismatch=True,
    )
    assert len(fake_grpc["secure"]) == 1


def test_bypass_flag_skips_identity_and_warns(fake_grpc):
    with pytest.warns(UserWarning, match="will NOT be verified"):
        create_grpc_channel(
            "any-host:50051",
            use_ssl=True,
            insecure_skip_server_identity_check=True,
        )
    assert len(fake_grpc["secure"]) == 1
    _, options = fake_grpc["secure"][0]
    # No SAN pin must be present when skip is on.
    keys = [k for k, _ in options]
    assert "grpc.ssl_target_name_override" not in keys


def test_invalid_server_hostname_rejected(fake_grpc):
    with pytest.raises(ValueError, match="wildcard"):
        create_grpc_channel(
            "localhost:50051",
            use_ssl=True,
            server_hostname="*.example.com",
        )
    with pytest.raises(ValueError, match="not a valid"):
        create_grpc_channel(
            "localhost:50051",
            use_ssl=True,
            server_hostname="bad_underscore.example",
        )


def test_insecure_channel_path_unchanged(fake_grpc):
    """server_hostname is irrelevant for insecure channels (no TLS to pin)."""
    create_grpc_channel("localhost:50051", use_ssl=False)
    assert len(fake_grpc["insecure"]) == 1
    assert fake_grpc["secure"] == []


# ---------------------------------------------------------------------------
# Live TLS handshake against a tiny gRPC server, using setup_ssl's generator
# to make end-to-end certs.
# ---------------------------------------------------------------------------


def _mint_cert(
    *,
    issuer: Tuple | None = None,
    dns_names: list[str] = None,
    ip_addresses: list[str] = None,
    is_ca: bool = False,
    common_name: str = "test",
):
    """Mint a self-signed CA or a CA-signed leaf cert with the requested SAN.

    Returns ``(private_key, certificate)``.
    """
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        ]
    )
    now = datetime.datetime.now(datetime.timezone.utc)
    if issuer is None:
        # self-signed
        issuer_name = subject
        signing_key = key
    else:
        issuer_key, issuer_cert = issuer
        issuer_name = issuer_cert.subject
        signing_key = issuer_key

    builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer_name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=1))
    )
    if is_ca:
        builder = builder.add_extension(
            x509.BasicConstraints(ca=True, path_length=0), critical=True
        )

    sans: list[x509.GeneralName] = []
    for d in dns_names or []:
        sans.append(x509.DNSName(d))
    for ip in ip_addresses or []:
        sans.append(x509.IPAddress(ipaddress.ip_address(ip)))
    if sans:
        builder = builder.add_extension(
            x509.SubjectAlternativeName(sans), critical=False
        )

    cert = builder.sign(private_key=signing_key, algorithm=hashes.SHA256())
    return key, cert


def _pem(key_or_cert, *, is_key: bool) -> bytes:
    from cryptography.hazmat.primitives import serialization

    if is_key:
        return key_or_cert.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    return key_or_cert.public_bytes(serialization.Encoding.PEM)


@pytest.fixture
def tls_fixture(tmp_path):
    """Generate a CA + matching server cert (SAN: localhost, 127.0.0.1) on
    disk, plus a server cert with the wrong SAN for negative tests."""
    ca_key, ca_cert = _mint_cert(is_ca=True, common_name="Test CA")
    good_key, good_cert = _mint_cert(
        issuer=(ca_key, ca_cert),
        dns_names=["localhost"],
        ip_addresses=["127.0.0.1"],
        common_name="localhost",
    )
    bad_key, bad_cert = _mint_cert(
        issuer=(ca_key, ca_cert),
        dns_names=["other.example.com"],
        common_name="other.example.com",
    )

    return {
        "ca_pem": _pem(ca_cert, is_key=False),
        "good_key_pem": _pem(good_key, is_key=True),
        "good_cert_pem": _pem(good_cert, is_key=False),
        "bad_key_pem": _pem(bad_key, is_key=True),
        "bad_cert_pem": _pem(bad_cert, is_key=False),
    }


def _start_server(server_key_pem: bytes, server_cert_pem: bytes, ca_pem: bytes):
    """Start a tiny gRPC server that just listens. The handshake is what's
    under test; we never call an RPC method."""
    import grpc
    from concurrent import futures

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    creds = grpc.ssl_server_credentials(
        ((server_key_pem, server_cert_pem),),
        root_certificates=ca_pem,
    )
    port = server.add_secure_port("127.0.0.1:0", creds)
    server.start()
    return server, port


def test_san_match_handshake_succeeds(tls_fixture, monkeypatch):
    # Use real grpc (un-patch fake_grpc fixture — not requested here).
    import grpc

    server, port = _start_server(
        tls_fixture["good_key_pem"],
        tls_fixture["good_cert_pem"],
        tls_fixture["ca_pem"],
    )
    try:
        channel = create_grpc_channel(
            f"127.0.0.1:{port}",
            use_ssl=True,
            root_certificate=tls_fixture["ca_pem"],
            server_hostname="localhost",
        )
        # channel_ready returns when the handshake completes successfully.
        grpc.channel_ready_future(channel).result(timeout=15)
        channel.close()
    finally:
        server.stop(0)


def test_san_mismatch_handshake_fails(tls_fixture):
    """Server cert has SAN=other.example.com; client pins localhost. Handshake
    must fail at TLS layer rather than letting any application data through."""
    import grpc

    server, port = _start_server(
        tls_fixture["bad_key_pem"],
        tls_fixture["bad_cert_pem"],
        tls_fixture["ca_pem"],
    )
    try:
        channel = create_grpc_channel(
            f"127.0.0.1:{port}",
            use_ssl=True,
            root_certificate=tls_fixture["ca_pem"],
            server_hostname="localhost",
        )
        with pytest.raises((grpc.FutureTimeoutError, grpc.RpcError)):
            grpc.channel_ready_future(channel).result(timeout=5)
        channel.close()
    finally:
        server.stop(0)
