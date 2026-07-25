"""Tests for src/appfl/comm/grpc/setup_ssl.py.

Covers the fixes for security-review findings #6 (shell-metachar guard on
ssl_dir — regression test only; no shell is invoked any more), #10 (CA private
key encrypted with a passphrase by default), and #16 (subject-field /
SAN-field validation, structural removal of the shell template).
"""

from __future__ import annotations

import os
import stat
import sys

import pytest

import appfl.comm.grpc.setup_ssl  # noqa: F401  (ensures submodule is loaded)
from appfl.comm.grpc.setup_ssl import setup_ssl

# The grpc package's __init__.py re-exports `setup_ssl` (function) under the
# same name as the submodule, shadowing it on the package object. Reach the
# actual module via sys.modules so monkeypatching attributes works.
setup_ssl_module = sys.modules["appfl.comm.grpc.setup_ssl"]


def _input_feeder(inputs):
    """Return a callable suitable for monkeypatching builtins.input that
    yields the given strings in order."""
    it = iter(inputs)

    def _input(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise AssertionError(
                f"setup_ssl asked for more input than the test provided. Last prompt: {prompt!r}"
            )

    return _input


@pytest.fixture
def env_passphrase(monkeypatch):
    """Run setup_ssl non-interactively with a fixed CA passphrase."""
    monkeypatch.setenv("APPFL_CA_PASSPHRASE", "test-passphrase-12")
    monkeypatch.delenv("APPFL_CA_NO_ENCRYPT", raising=False)
    return b"test-passphrase-12"


def _default_subject_inputs():
    """Five empty strings: accept all subject-field defaults (C, ST, ORG, DNS, IP)."""
    return ["", "", "", "", ""]


def _run_setup(monkeypatch, ssl_dir, *, extra_inputs=()):
    """Drive setup_ssl with one ssl_dir + the five default subject fields."""
    inputs = [str(ssl_dir), *_default_subject_inputs(), *extra_inputs]
    monkeypatch.setattr("builtins.input", _input_feeder(inputs))
    setup_ssl()


# ---------------------------------------------------------------------------
# Regression tests for #6 (shell-metachar guard on ssl_dir).
# The shell script and os.system call were removed by #16, but the path
# whitelist must continue to reject every metachar class.
# ---------------------------------------------------------------------------


def test_rejects_shell_metachars_in_ssl_dir(
    monkeypatch, tmp_path, env_passphrase, capsys
):
    pwned_marker = tmp_path / "pwned"
    good_dir = tmp_path / "ssl"

    malicious = f"/tmp/x$(touch {pwned_marker})"

    monkeypatch.setattr(
        "builtins.input",
        _input_feeder([malicious, str(good_dir), *_default_subject_inputs()]),
    )

    setup_ssl()

    assert not pwned_marker.exists(), (
        "Shell command substitution executed — the metachar guard is broken"
    )
    assert good_dir.is_dir()
    assert (good_dir / "ca.key").is_file()
    assert "Invalid directory" in capsys.readouterr().out


@pytest.mark.parametrize(
    "bad",
    [
        "/tmp/a;rm -rf /",
        "/tmp/a|whoami",
        "/tmp/a&echo x",
        "/tmp/a`id`",
        "/tmp/a$(id)",
        "/tmp/a\nid",
        "/tmp/a*",
        "/tmp/a?",
        "/tmp/a b",
        "/tmp/a>b",
    ],
)
def test_metachar_classes_each_rejected(
    monkeypatch, tmp_path, env_passphrase, capsys, bad
):
    good_dir = tmp_path / "ssl"
    monkeypatch.setattr(
        "builtins.input",
        _input_feeder([bad, str(good_dir), *_default_subject_inputs()]),
    )
    setup_ssl()
    assert good_dir.is_dir()
    assert "Invalid directory" in capsys.readouterr().out


def test_no_shell_invocation(monkeypatch, tmp_path, env_passphrase):
    """The rewrite removed all shell calls. Patch every shell entry point to
    fail loudly; the happy path must complete without touching any of them."""

    def _boom_os_system(cmd):  # pragma: no cover - asserted to never run
        raise AssertionError(
            f"setup_ssl invoked os.system({cmd!r}) — regression of #6/#16"
        )

    monkeypatch.setattr(setup_ssl_module.os, "system", _boom_os_system)

    if hasattr(setup_ssl_module, "subprocess"):
        # The module no longer imports subprocess; if it ever re-introduces
        # one, this guard fires on it.
        raise AssertionError(
            "setup_ssl re-imported subprocess; the rewrite intentionally "
            "removed the shell path."
        )

    _run_setup(monkeypatch, tmp_path / "ssl")


# ---------------------------------------------------------------------------
# #10 — CA private key is encrypted by default.
# ---------------------------------------------------------------------------


def _load_pem_key(path, password=None):
    from cryptography.hazmat.primitives import serialization

    with open(path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=password)


def test_ca_key_requires_passphrase(monkeypatch, tmp_path, env_passphrase):
    ssl_dir = tmp_path / "ssl"
    _run_setup(monkeypatch, ssl_dir)
    with pytest.raises(TypeError):
        _load_pem_key(ssl_dir / "ca.key", password=None)
    key = _load_pem_key(ssl_dir / "ca.key", password=env_passphrase)
    assert key.key_size == 4096


def test_ca_key_wrong_passphrase_rejected(monkeypatch, tmp_path, env_passphrase):
    ssl_dir = tmp_path / "ssl"
    _run_setup(monkeypatch, ssl_dir)
    with pytest.raises(ValueError):
        _load_pem_key(ssl_dir / "ca.key", password=b"wrong-passphrase")


def test_server_key_is_unencrypted(monkeypatch, tmp_path, env_passphrase):
    """gRPC has no passphrase callback for ssl_server_credentials, so the
    server key must remain loadable without a password — only the CA key is
    encrypted."""
    ssl_dir = tmp_path / "ssl"
    _run_setup(monkeypatch, ssl_dir)
    key = _load_pem_key(ssl_dir / "server.key", password=None)
    assert key.key_size == 4096


def test_ca_password_notsafe_regression_guard():
    """The 'CA_PASSWORD=notsafe' line was dead in the old bash template and
    has been removed entirely. Guard against re-introduction."""
    src = sys.modules["appfl.comm.grpc.setup_ssl"].__file__
    with open(src) as f:
        body = f.read()
    assert "CA_PASSWORD=notsafe" not in body
    assert "notsafe" not in body


def test_unencrypted_opt_out_via_env(monkeypatch, tmp_path):
    """APPFL_CA_NO_ENCRYPT=1 in a non-interactive run produces a plaintext
    key and writes a WARNING banner to stderr."""
    monkeypatch.delenv("APPFL_CA_PASSPHRASE", raising=False)
    monkeypatch.setenv("APPFL_CA_NO_ENCRYPT", "1")
    monkeypatch.setattr(setup_ssl_module.sys.stdin, "isatty", lambda: False)

    ssl_dir = tmp_path / "ssl"
    _run_setup(monkeypatch, ssl_dir)

    key = _load_pem_key(ssl_dir / "ca.key", password=None)
    assert key.key_size == 4096


def test_refuses_unattended_without_passphrase(monkeypatch, tmp_path):
    """Non-interactive with no passphrase and no APPFL_CA_NO_ENCRYPT must
    refuse rather than silently writing an unencrypted key."""
    monkeypatch.delenv("APPFL_CA_PASSPHRASE", raising=False)
    monkeypatch.delenv("APPFL_CA_NO_ENCRYPT", raising=False)
    monkeypatch.setattr(setup_ssl_module.sys.stdin, "isatty", lambda: False)

    ssl_dir = tmp_path / "ssl"
    monkeypatch.setattr(
        "builtins.input",
        _input_feeder([str(ssl_dir), *_default_subject_inputs()]),
    )
    with pytest.raises(RuntimeError, match="refusing"):
        setup_ssl()


# ---------------------------------------------------------------------------
# #16 — subject-field validation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field_idx,bad_value",
    [
        # Country code: must be exactly 2 letters
        (0, "USA"),
        (0, "1"),
        (0, "US; rm -rf"),
        # State: rejects shell metachars + newlines
        (1, 'Illinois"; echo x; echo "'),
        (1, "Illinois\nDNS.2 = evil"),
        (1, "Illinois\n[dn]\nCN = evil"),
        # Org: same class
        (2, "APPFL`id`"),
        (2, "APPFL$(id)"),
    ],
)
def test_subject_field_rejects_injection(
    monkeypatch, tmp_path, env_passphrase, capsys, field_idx, bad_value
):
    """For each subject field, a bad value must be rejected and the prompt
    re-asked. The fallback acceptable value finishes the flow.

    Note: an empty input is accepted (it means "use the default") and is not
    a rejection case — that's tested separately by the happy-path tests.
    """
    fallback_subject = _default_subject_inputs()
    inputs = (
        [str(tmp_path / "ssl")]
        + fallback_subject[:field_idx]
        + [bad_value]
        + fallback_subject[field_idx:]
    )
    monkeypatch.setattr("builtins.input", _input_feeder(inputs))
    setup_ssl()
    out = capsys.readouterr().out
    assert "Please try again" in out


@pytest.mark.parametrize(
    "bad_dns",
    [
        "-foo.example",
        "foo_bar.example",
        ".foo.example",
        "foo..example",
        "a" * 254,
        "label-too-long-" + "x" * 60 + ".example",
        "foo\n[dn]\nCN=evil",
        "foo; rm",
    ],
)
def test_dns_field_validation(monkeypatch, tmp_path, env_passphrase, capsys, bad_dns):
    """DNS field rejects RFC-1123-invalid hostnames and injection payloads."""
    fallback = _default_subject_inputs()
    # Subject inputs: C, ST, ORG, DNS, IP. DNS is index 3.
    inputs = [str(tmp_path / "ssl")] + fallback[:3] + [bad_dns] + fallback[3:]
    monkeypatch.setattr("builtins.input", _input_feeder(inputs))
    setup_ssl()
    assert "try again" in capsys.readouterr().out.lower()


@pytest.mark.parametrize(
    "bad_ip",
    [
        "999.999.999.999",
        "127.0.0.1; rm -rf /",
        "127.0.0.1\nDNS.2 = evil",
        "localhost",
    ],
)
def test_ip_field_validation(monkeypatch, tmp_path, env_passphrase, capsys, bad_ip):
    fallback = _default_subject_inputs()
    # IP is index 4.
    inputs = [str(tmp_path / "ssl")] + fallback[:4] + [bad_ip] + fallback[4:]
    monkeypatch.setattr("builtins.input", _input_feeder(inputs))
    setup_ssl()
    assert "try again" in capsys.readouterr().out.lower()


# ---------------------------------------------------------------------------
# #9 — multi-SAN and resulting cert validity.
# ---------------------------------------------------------------------------


def test_multi_san_dns_and_ip(monkeypatch, tmp_path, env_passphrase):
    """Comma-separated DNS and IP inputs land in the cert SAN."""
    ssl_dir = tmp_path / "ssl"
    inputs = [
        str(ssl_dir),
        "",  # C
        "",  # ST
        "",  # ORG
        "appfl.example.com, alt.example.com, localhost",  # DNS
        "10.0.0.5, 127.0.0.1",  # IP
    ]
    monkeypatch.setattr("builtins.input", _input_feeder(inputs))
    setup_ssl()

    from cryptography import x509

    with open(ssl_dir / "server.crt", "rb") as f:
        cert = x509.load_pem_x509_certificate(f.read())
    san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert sorted(san.get_values_for_type(x509.DNSName)) == [
        "alt.example.com",
        "appfl.example.com",
        "localhost",
    ]
    assert sorted(str(ip) for ip in san.get_values_for_type(x509.IPAddress)) == [
        "10.0.0.5",
        "127.0.0.1",
    ]


def test_server_cert_is_signed_by_ca(monkeypatch, tmp_path, env_passphrase):
    ssl_dir = tmp_path / "ssl"
    _run_setup(monkeypatch, ssl_dir)

    from cryptography import x509
    from cryptography.hazmat.primitives.asymmetric import padding

    with open(ssl_dir / "ca.crt", "rb") as f:
        ca = x509.load_pem_x509_certificate(f.read())
    with open(ssl_dir / "server.crt", "rb") as f:
        leaf = x509.load_pem_x509_certificate(f.read())

    assert leaf.issuer == ca.subject

    # Verify the signature on the leaf was produced by the CA's key.
    ca.public_key().verify(
        leaf.signature,
        leaf.tbs_certificate_bytes,
        padding.PKCS1v15(),
        leaf.signature_hash_algorithm,
    )


def test_ca_cert_has_basic_constraints_and_key_usage(
    monkeypatch, tmp_path, env_passphrase
):
    ssl_dir = tmp_path / "ssl"
    _run_setup(monkeypatch, ssl_dir)
    from cryptography import x509

    with open(ssl_dir / "ca.crt", "rb") as f:
        ca = x509.load_pem_x509_certificate(f.read())
    bc = ca.extensions.get_extension_for_class(x509.BasicConstraints).value
    assert bc.ca is True
    ku = ca.extensions.get_extension_for_class(x509.KeyUsage).value
    assert ku.key_cert_sign is True
    assert ku.crl_sign is True


# ---------------------------------------------------------------------------
# Filesystem hardening.
# ---------------------------------------------------------------------------


def test_file_modes(monkeypatch, tmp_path, env_passphrase):
    ssl_dir = tmp_path / "ssl"
    old_umask = os.umask(0o022)
    try:
        _run_setup(monkeypatch, ssl_dir)
    finally:
        os.umask(old_umask)
    assert stat.S_IMODE(os.stat(ssl_dir).st_mode) == 0o700
    for name in ("ca.key", "server.key"):
        m = stat.S_IMODE(os.stat(ssl_dir / name).st_mode)
        assert m == 0o600, f"{name}: expected 0o600, got {oct(m)}"
    for name in ("ca.crt", "server.crt"):
        m = stat.S_IMODE(os.stat(ssl_dir / name).st_mode)
        assert m == 0o644, f"{name}: expected 0o644, got {oct(m)}"


def test_refuses_symlink_at_ssl_dir(monkeypatch, tmp_path, env_passphrase, capsys):
    """If the target ssl_dir is a symlink, refuse rather than follow it."""
    real_target = tmp_path / "real"
    real_target.mkdir(mode=0o700)
    symlink_dir = tmp_path / "via-symlink"
    os.symlink(real_target, symlink_dir)

    good_dir = tmp_path / "ssl"
    monkeypatch.setattr(
        "builtins.input",
        _input_feeder([str(symlink_dir), str(good_dir), *_default_subject_inputs()]),
    )
    setup_ssl()
    out = capsys.readouterr().out
    assert "symlink" in out.lower() or "Invalid directory" in out
    assert good_dir.is_dir()


def test_refuses_group_writable_ssl_dir(monkeypatch, tmp_path, env_passphrase, capsys):
    bad = tmp_path / "loose"
    bad.mkdir(mode=0o770)
    os.chmod(bad, 0o770)
    good = tmp_path / "ssl"
    monkeypatch.setattr(
        "builtins.input",
        _input_feeder([str(bad), str(good), *_default_subject_inputs()]),
    )
    setup_ssl()
    out = capsys.readouterr().out
    assert (
        "group" in out.lower()
        or "world-writable" in out.lower()
        or "Invalid directory" in out
    )
    assert good.is_dir()
