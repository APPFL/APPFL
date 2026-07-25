# SSL/TLS Credentials

This directory used to ship a `localhost.crt` / `localhost.key` / `root.crt`
trio borrowed from the gRPC examples, intended for local demos. Those files
have been removed: shipping a private key in a PyPI wheel is a supply-chain
hazard (every `pip install appfl` user gets the same key on disk, and any
on-path attacker can MITM `localhost` deployments using the matching public
certificate).

## Generating credentials

Use the bundled console script to generate a fresh local CA and a server
certificate signed by it:

```bash
appfl-setup-ssl
```

The script writes (by default) to `~/.appfl/ssl/`:

- `ca.key` and `ca.crt` — the CA private key and self-signed certificate.
  Keep `ca.key` private; ship `ca.crt` to every client that should trust
  this federation.
- `server.key` and `server.crt` — the server's private key and the CA-signed
  certificate to present on the wire.

The CA private key is **encrypted with a passphrase** by default. The script
prompts for the passphrase interactively and applies AES-256 / PBKDF2 via
PKCS#8 (`cryptography`'s `BestAvailableEncryption`). Anyone who reads
`ca.key` off the disk still needs the passphrase to forge certificates.

### Scripted / CI use

Set `APPFL_CA_PASSPHRASE` in the environment to suppress the prompt:

```bash
APPFL_CA_PASSPHRASE='use-a-real-passphrase' appfl-setup-ssl < answers.txt
```

To deliberately generate an unencrypted CA key (CI-only, do not do this on
shared hosts), set `APPFL_CA_NO_ENCRYPT=1`. The script writes a loud WARNING
banner to stderr and refuses to proceed without an interactive "yes"
confirmation when run on a TTY.

### Multi-SAN certificates

The DNS and IP prompts accept comma-separated lists, so one server cert can
cover the bind IP, the hostname, and `localhost` at once:

```text
Enter DNS name(s), comma-separated, press Enter to use default 'localhost': appfl.example.com, internal-lb.example.com, localhost
Enter IP address(es), comma-separated, press Enter to use default '127.0.0.1': 10.0.0.5, 127.0.0.1
```

After generation the script prints the `server_hostname` value clients must
pin (the first DNS entry) and the full list of SAN entries the cert covers.

## Pointing APPFL at the generated paths

```yaml
server:
  server_certificate_key: /home/<you>/.appfl/ssl/server.key
  server_certificate:     /home/<you>/.appfl/ssl/server.crt
client:
  use_ssl: true
  root_certificate:       /home/<you>/.appfl/ssl/ca.crt
  server_hostname:        appfl.example.com   # must match a SAN on server.crt
```

`server_hostname` is **required** when `use_ssl: true`. It is the identity
the client expects the server to prove, independent of the URI the client
dials. Without it, any certificate that chains to the configured CA would be
accepted regardless of subject — an attacker who can mint a leaf cert from
the same CA could MITM the connection.

In code, paths are loaded via `appfl.comm.grpc.load_credential_from_file`.

## Production deployments

`appfl-setup-ssl` is a convenience for self-signed local CAs. For production,
mint server certificates from a real CA the participating clients already
trust (your institutional PKI, Let's Encrypt for an internet-facing endpoint,
an AWS PCA, etc.) and configure the same fields to point at those PEM files.
Set `server_hostname` to the SAN you expect to see on the production cert.

## Rotating the CA

To rotate the CA (e.g. because `ca.key` was exposed):

1. Run `appfl-setup-ssl` to generate a fresh CA and server cert.
2. Distribute the new `ca.crt` to every client out of band.
3. Restart the server with the new `server.key` / `server.crt`.

There is no in-band revocation path; rotation is the only remedy.
