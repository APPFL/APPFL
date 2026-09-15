"""Generate a self-signed CA and a CA-signed server certificate for an APPFL
gRPC deployment.

This module is the implementation of the ``appfl-setup-ssl`` console script.
The generator is implemented directly with the :mod:`cryptography` library
rather than by templating a Bash script that shells out to ``openssl``. The
direct approach removes three classes of risk that the previous template had:

* Shell metacharacters in the operator-supplied subject fields (``C``, ``ST``,
  ``ORG``, ``DNS``, ``IP``) cannot break out of the ``openssl`` argv because
  there is no shell — every value flows through a typed :mod:`cryptography`
  call.
* The CA private key is encrypted with a passphrase by default. Reading
  ``ca.key`` off a shared filesystem is no longer sufficient to forge certs
  for the federation.
* The wizard accepts comma-separated DNS and IP inputs, so a single server
  certificate can cover the bind IP, the hostname, and a localhost alias at
  once. The expected ``server_hostname`` values that clients must pin are
  printed at the end of the run.
"""

from __future__ import annotations

import datetime
import getpass
import ipaddress
import os
import pathlib
import re
import stat
import sys

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

_SAFE_PATH_RE = re.compile(r"^/[A-Za-z0-9_./-]*$")

# ISO 3166-1 alpha-2 country code (two uppercase letters).
_RE_COUNTRY = re.compile(r"^[A-Za-z]{2}$")

# State / province / organisation: printable ASCII, no shell metachars, no
# newlines, no `[` (which could otherwise inject an X.509 attribute split).
_RE_NAME = re.compile(r"^[A-Za-z0-9 .,'_\-]{1,64}$")

# RFC 1123-ish hostname: dot-separated labels, each label 1-63 chars, total
# <=253 chars, no leading/trailing dash on any label. Underscores are rejected.
_RE_DNS = re.compile(
    r"^(?=.{1,253}$)"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?)"
    r"(?:\.(?:[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?))*$"
)

# Minimum CA passphrase length when prompted interactively. Operators on the
# scripted path (env var) can set whatever they want.
_MIN_PASSPHRASE_LEN = 12

# Validity windows.
_CA_VALIDITY_DAYS = 3650
_LEAF_VALIDITY_DAYS = 825


def _prompt_ssl_dir() -> str:
    default_ssl_dir = os.path.join(pathlib.Path.home(), ".appfl", "ssl")
    while True:
        ssl_dir = input(
            f"Enter the absolute path of the directory where the SSL certificate and private key will be stored, press Enter to use the default directory {default_ssl_dir}: "
        )
        if not ssl_dir:
            ssl_dir = default_ssl_dir
        ssl_dir = os.path.abspath(os.path.expanduser(ssl_dir))
        if not _SAFE_PATH_RE.match(ssl_dir):
            print(
                "Invalid directory: only absolute paths containing letters, "
                "digits, '.', '_', '-', and '/' are allowed. Please try again."
            )
            continue
        try:
            _ensure_ssl_dir(ssl_dir)
        except OSError as e:
            print(f"Invalid directory ({e}), please try again")
            continue
        return ssl_dir


def _ensure_ssl_dir(ssl_dir: str) -> None:
    """Create ``ssl_dir`` if missing, refuse symlinks and group/world writes,
    and clear any pre-existing regular files. Mode is forced to 0o700."""
    if os.path.lexists(ssl_dir):
        # Reject any symlink at the target path so a co-tenant can't redirect
        # the writes via a pre-positioned symlink.
        st = os.lstat(ssl_dir)
        if stat.S_ISLNK(st.st_mode):
            raise OSError(
                f"{ssl_dir} is a symlink; refusing to follow it. "
                "Remove the symlink and try again."
            )
        if not stat.S_ISDIR(st.st_mode):
            raise NotADirectoryError(ssl_dir)
        if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise OSError(
                f"{ssl_dir} is group- or world-writable; refusing to use it. "
                "Run `chmod 700` on the directory and try again."
            )
        # Clear leftover files, but refuse to follow symlinks inside.
        for name in os.listdir(ssl_dir):
            entry = os.path.join(ssl_dir, name)
            entry_st = os.lstat(entry)
            if stat.S_ISDIR(entry_st.st_mode) and not stat.S_ISLNK(entry_st.st_mode):
                # Don't recurse into subdirectories — operator put them there.
                continue
            os.unlink(entry)
    else:
        os.makedirs(ssl_dir, mode=0o700, exist_ok=False)
    # `mkdir` honours umask, so chmod explicitly.
    os.chmod(ssl_dir, 0o700)


def _validate(field: str, value: str, pattern: re.Pattern) -> str:
    value = value.strip()
    if not pattern.match(value):
        raise ValueError(
            f"Invalid {field} ({value!r}): must match {pattern.pattern!r}. "
            "Newlines, quotes, and shell metacharacters are rejected."
        )
    return value


def _validate_country(value: str) -> str:
    return _validate("Country (C)", value, _RE_COUNTRY).upper()


def _validate_name(field: str, value: str) -> str:
    return _validate(field, value, _RE_NAME)


def _validate_dns(value: str) -> str:
    return _validate("DNS", value, _RE_DNS)


def _validate_ip(value: str) -> str:
    # `ipaddress.ip_address` rejects everything the regex would have to
    # enumerate manually: "127.0.0.1; rm", "999.999.999.999", newlines, etc.
    value = value.strip()
    return str(ipaddress.ip_address(value))


def _prompt_validated(prompt: str, default: str, validator) -> str:
    """Interactively prompt until ``validator`` accepts the value. The default
    is used on a bare Enter and is also validated (so the wizard fails loudly
    if a hard-coded default ever becomes invalid)."""
    while True:
        raw = input(prompt) or default
        try:
            return validator(raw)
        except ValueError as e:
            print(f"  {e}. Please try again.")


def _split_csv(raw: str) -> list[str]:
    """Split a comma-separated input into trimmed non-empty parts."""
    return [part.strip() for part in raw.split(",") if part.strip()]


def _prompt_san_dns(default: str) -> list[str]:
    while True:
        raw = (
            input(
                f"Enter DNS name(s), comma-separated, press Enter to use default '{default}': "
            )
            or default
        )
        parts = _split_csv(raw)
        if not parts:
            print("  At least one DNS name is required. Please try again.")
            continue
        try:
            return [_validate_dns(p) for p in parts]
        except ValueError as e:
            print(f"  {e}. Please try again.")


def _prompt_san_ip(default: str) -> list[str]:
    while True:
        raw = (
            input(
                f"Enter IP address(es), comma-separated, press Enter to use default '{default}': "
            )
            or default
        )
        parts = _split_csv(raw)
        if not parts:
            print("  At least one IP address is required. Please try again.")
            continue
        try:
            return [_validate_ip(p) for p in parts]
        except ValueError as e:
            print(f"  {e}. Please try again.")


def _acquire_ca_passphrase() -> bytes | None:
    """Return the CA private-key passphrase as bytes, or ``None`` if the
    operator explicitly opted out of encryption.

    Order of precedence:

    1. ``APPFL_CA_PASSPHRASE`` env var (scripted / CI path).
    2. Interactive double-prompt via :func:`getpass.getpass` when stdin is a
       TTY.
    3. ``APPFL_CA_NO_ENCRYPT=1`` env var (loud opt-out for unattended runs).
    """
    env_pw = os.environ.get("APPFL_CA_PASSPHRASE")
    if env_pw:
        return env_pw.encode("utf-8")

    no_encrypt = os.environ.get("APPFL_CA_NO_ENCRYPT", "").strip() in (
        "1",
        "true",
        "True",
        "yes",
    )

    if not sys.stdin.isatty():
        if no_encrypt:
            _warn_unencrypted_ca_key()
            return None
        raise RuntimeError(
            "appfl-setup-ssl: refusing to generate an unencrypted CA private "
            "key in a non-interactive shell. Set APPFL_CA_PASSPHRASE=... to "
            "encrypt it, or APPFL_CA_NO_ENCRYPT=1 to acknowledge the risk."
        )

    for _ in range(3):
        first = getpass.getpass(
            f"CA private key passphrase (>= {_MIN_PASSPHRASE_LEN} chars, empty = unencrypted): "
        )
        if first == "":
            if no_encrypt or _confirm_unencrypted_interactive():
                _warn_unencrypted_ca_key()
                return None
            continue
        if len(first) < _MIN_PASSPHRASE_LEN:
            print(
                f"  Passphrase too short (need >= {_MIN_PASSPHRASE_LEN} chars). "
                "Please try again."
            )
            continue
        second = getpass.getpass("Confirm passphrase: ")
        if first != second:
            print("  Passphrases did not match. Please try again.")
            continue
        return first.encode("utf-8")

    raise RuntimeError(
        "appfl-setup-ssl: gave up after three failed passphrase attempts."
    )


def _confirm_unencrypted_interactive() -> bool:
    print(
        "\n  WARNING: an unencrypted CA private key is the federation's root "
        "of trust on disk.\n"
        "  Anyone who reads ca.key can mint server or client certificates for "
        "any identity.\n"
    )
    answer = input("  Type 'yes' to proceed without encryption: ").strip().lower()
    return answer == "yes"


def _warn_unencrypted_ca_key() -> None:
    banner = "!" * 72
    print(banner, file=sys.stderr)
    print(
        "WARNING: ca.key is being written UNENCRYPTED. Anyone who can read "
        "this file can mint certificates for any identity in the federation.",
        file=sys.stderr,
    )
    print(
        "To encrypt an existing key after the fact:\n"
        "    openssl pkcs8 -topk8 -v2 aes-256-cbc -in ca.key -out ca.key.enc\n"
        "    mv ca.key.enc ca.key",
        file=sys.stderr,
    )
    print(banner, file=sys.stderr)


def _build_name(country: str, state: str, org: str, common_name: str) -> x509.Name:
    return x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, country),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, state),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, org),
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        ]
    )


def _build_san(
    dns_names: list[str], ip_addresses: list[str]
) -> x509.SubjectAlternativeName:
    entries: list[x509.GeneralName] = [x509.DNSName(d) for d in dns_names]
    entries.extend(x509.IPAddress(ipaddress.ip_address(ip)) for ip in ip_addresses)
    return x509.SubjectAlternativeName(entries)


def _utcnow() -> datetime.datetime:
    # Backwards-compatible across cryptography versions: aware UTC datetime
    # is required by current cryptography releases and accepted by older ones.
    return datetime.datetime.now(datetime.timezone.utc)


def _generate_ca(
    name: x509.Name, key_size: int = 4096
) -> tuple[rsa.RSAPrivateKey, x509.Certificate]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=key_size)
    now = _utcnow()
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=_CA_VALIDITY_DAYS))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=False,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(key.public_key()),
            critical=False,
        )
        .sign(private_key=key, algorithm=hashes.SHA256())
    )
    return key, cert


def _generate_server_cert(
    *,
    ca_key: rsa.RSAPrivateKey,
    ca_cert: x509.Certificate,
    subject: x509.Name,
    san: x509.SubjectAlternativeName,
    key_size: int = 4096,
) -> tuple[rsa.RSAPrivateKey, x509.Certificate]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=key_size)
    now = _utcnow()
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=_LEAF_VALIDITY_DAYS))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage(
                [
                    x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
                ]
            ),
            critical=False,
        )
        .add_extension(san, critical=False)
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(key.public_key()),
            critical=False,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_cert.public_key()),
            critical=False,
        )
        .sign(private_key=ca_key, algorithm=hashes.SHA256())
    )
    return key, cert


def _write_private_key(
    path: str, key: rsa.RSAPrivateKey, passphrase: bytes | None
) -> None:
    if passphrase:
        encryption: serialization.KeySerializationEncryption = (
            serialization.BestAvailableEncryption(passphrase)
        )
    else:
        encryption = serialization.NoEncryption()
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=encryption,
    )
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(pem)
    except Exception:
        os.close(fd)
        raise
    os.chmod(path, 0o600)


def _write_certificate(path: str, cert: x509.Certificate) -> None:
    pem = cert.public_bytes(serialization.Encoding.PEM)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o644)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(pem)
    except Exception:
        os.close(fd)
        raise
    os.chmod(path, 0o644)


def setup_ssl():
    """Console entry point for ``appfl-setup-ssl``.

    Prompts the operator for a target directory and X.509 subject fields,
    generates a self-signed CA and a CA-signed server certificate, and writes
    them under the target directory. The CA private key is encrypted with a
    passphrase by default — set ``APPFL_CA_PASSPHRASE`` to script the prompt,
    or ``APPFL_CA_NO_ENCRYPT=1`` (plus an interactive confirmation, when on a
    TTY) to opt out.
    """
    ssl_dir = _prompt_ssl_dir()

    default_C = "US"
    default_ST = "Illinois"
    default_O = "APPFL"
    default_DNS = "localhost"
    default_IP = "127.0.0.1"

    country = _prompt_validated(
        f"Enter Country Code (2 letters), press Enter to use default '{default_C}': ",
        default_C,
        _validate_country,
    )
    state = _prompt_validated(
        f"Enter State, press Enter to use default '{default_ST}': ",
        default_ST,
        lambda v: _validate_name("State (ST)", v),
    )
    org = _prompt_validated(
        f"Enter Organization (O), press Enter to use default '{default_O}': ",
        default_O,
        lambda v: _validate_name("Organization (O)", v),
    )
    dns_names = _prompt_san_dns(default_DNS)
    ip_addresses = _prompt_san_ip(default_IP)

    # The CN is informational under modern X.509 verifiers (which look at the
    # SAN). Use the first DNS name so the cert still chains nicely on legacy
    # tooling that consults the subject CN.
    common_name = dns_names[0]

    passphrase = _acquire_ca_passphrase()

    ca_subject = _build_name(country, state, org, f"{org} Root CA")
    ca_key, ca_cert = _generate_ca(ca_subject)

    server_subject = _build_name(country, state, org, common_name)
    server_san = _build_san(dns_names, ip_addresses)
    server_key, server_cert = _generate_server_cert(
        ca_key=ca_key,
        ca_cert=ca_cert,
        subject=server_subject,
        san=server_san,
    )

    ca_key_path = os.path.join(ssl_dir, "ca.key")
    ca_crt_path = os.path.join(ssl_dir, "ca.crt")
    server_key_path = os.path.join(ssl_dir, "server.key")
    server_crt_path = os.path.join(ssl_dir, "server.crt")

    _write_private_key(ca_key_path, ca_key, passphrase)
    _write_certificate(ca_crt_path, ca_cert)
    _write_private_key(server_key_path, server_key, passphrase=None)
    _write_certificate(server_crt_path, server_cert)

    rule = "=" * 78
    print(rule)
    print(f"CA certificate stored in  {ca_crt_path}")
    print(
        f"CA private key stored in  {ca_key_path}"
        + (" (encrypted)" if passphrase else " (UNENCRYPTED)")
    )
    print(f"Server certificate stored in {server_crt_path}")
    print(f"Server private key stored in {server_key_path}")
    print(rule)
    print(f"Copy {ca_crt_path} to every client that should trust this federation.")
    print(
        "On each client, set in the gRPC client communicator config:\n"
        f"  root_certificate: {ca_crt_path}\n"
        "  use_ssl: true"
    )
    print(
        "Pin the server identity (must match a SAN below) using:\n"
        f"  server_hostname: {dns_names[0]}"
    )
    print(
        "Subject alternative names on this server cert:\n"
        f"  DNS: {', '.join(dns_names)}\n"
        f"  IP:  {', '.join(ip_addresses)}"
    )
    print(rule)
