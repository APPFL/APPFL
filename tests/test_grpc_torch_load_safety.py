"""Tests for safe ``torch.load`` use on the gRPC path.

Covers three layered changes:

1. :func:`appfl.comm.grpc.utils.deserialize_model` and the two private
   deserialisers (``ServerAgent._bytes_to_model``,
   ``GRPCClientCommunicator._deserialize_model_optimized``) all pass
   ``weights_only=True`` to ``torch.load``. A pickle gadget embedded in the
   payload must raise rather than execute.

2. ``GRPCServerCommunicator`` refuses ``_use_proxystore`` /
   ``_use_colab_connector`` / ``_use_s3`` metadata flags from the client
   unless the corresponding server-side feature is enabled, so a malicious
   client cannot opt the server into the only branches that still need
   ``weights_only=False``.

3. ``validate_state_dict`` enforces the federation's actual schema (keys,
   shapes, dtypes) on whatever was deserialised. Even payloads that survive
   the weights-only unpickler must match the server's model, so a client
   cannot inject arbitrary tensors / objects / extra parameters.
"""

from __future__ import annotations

import io

import grpc
import pytest
import torch

import appfl.comm.grpc.utils as grpc_utils
from appfl.comm.grpc.utils import deserialize_model, serialize_model


# ---------------------------------------------------------------------------
# (1) weights_only=True on the deserialisers.
# ---------------------------------------------------------------------------


_GADGET_MARKER = {"executed": False}


def _fire_gadget():
    """Module-level so pickle can address it. Sets a marker if it ever runs."""
    _GADGET_MARKER["executed"] = True
    return "gadget-return"


class _PickleGadget:
    """Object whose unpickling executes :func:`_fire_gadget`. Used as a
    positive control: if ``torch.load`` runs Python at deserialise time, the
    side effect on the marker dict happens."""

    def __reduce__(self):
        return (_fire_gadget, ())


def _gadget_bytes() -> bytes:
    """Produce torch-saved bytes that ``torch.load(weights_only=False)``
    would execute and ``torch.load(weights_only=True)`` must refuse.

    We wrap the gadget in a torch.save container so the bytes are a valid
    torch checkpoint envelope — what an attacker would actually send. Bare
    ``pickle.dumps`` lacks the magic number and would fail before the
    weights_only check.
    """
    _GADGET_MARKER["executed"] = False
    buf = io.BytesIO()
    torch.save({"evil": _PickleGadget()}, buf)
    return buf.getvalue()


def test_deserialize_model_defaults_to_weights_only():
    """Sanity-check the default kwarg — the whole PR rests on this default."""
    import inspect

    params = inspect.signature(grpc_utils.deserialize_model).parameters
    assert params["weights_only"].default is True, (
        "deserialize_model must default to weights_only=True; "
        f"got {params['weights_only'].default!r}"
    )


def test_deserialize_model_rejects_pickle_gadget():
    """A pickle gadget shipped as a fake 'model' must NOT execute."""
    payload = _gadget_bytes()
    with pytest.raises(Exception):
        deserialize_model(payload)
    assert _GADGET_MARKER["executed"] is False, (
        "pickle gadget executed during deserialize_model — weights_only is "
        "not being enforced"
    )


def test_deserialize_model_round_trips_state_dict():
    """The happy path: a torch state dict round-trips through
    serialize_model -> deserialize_model unchanged."""
    state = {
        "linear.weight": torch.randn(3, 4),
        "linear.bias": torch.zeros(4),
        "step": torch.tensor(7, dtype=torch.int64),
    }
    blob = serialize_model(state)
    out = deserialize_model(blob)
    assert set(out.keys()) == set(state.keys())
    for k in state:
        assert torch.equal(out[k], state[k])


def test_deserialize_model_explicit_weights_only_false_runs_pickle():
    """Smoke test of the escape hatch: passing weights_only=False explicitly
    opts back into the unsafe codec. Guards against the kwarg being
    silently dropped by a future refactor."""
    payload = _gadget_bytes()
    deserialize_model(payload, weights_only=False)
    assert _GADGET_MARKER["executed"] is True


def test_server_bytes_to_model_uses_weights_only(monkeypatch):
    """ServerAgent._bytes_to_model must refuse a pickle gadget too."""
    from appfl.agent.server import ServerAgent

    # Build a minimal ServerAgent without going through the full config path:
    # only _bytes_to_model is under test and it only depends on two flags.
    agent = ServerAgent.__new__(ServerAgent)
    agent.enable_compression = False
    agent.optimize_memory = True

    payload = _gadget_bytes()
    with pytest.raises(Exception):
        agent._bytes_to_model(payload)
    assert _GADGET_MARKER["executed"] is False

    # And again on the non-optimize_memory branch.
    agent.optimize_memory = False
    with pytest.raises(Exception):
        agent._bytes_to_model(_gadget_bytes())
    assert _GADGET_MARKER["executed"] is False


def test_client_deserialize_uses_weights_only(monkeypatch):
    """GRPCClientCommunicator._deserialize_model_optimized refuses a gadget.

    The client communicator's __init__ opens a real gRPC channel, so we
    construct the object without calling __init__ and exercise the bound
    method directly.
    """
    from appfl.comm.grpc.grpc_client_communicator import GRPCClientCommunicator

    client = GRPCClientCommunicator.__new__(GRPCClientCommunicator)
    payload = _gadget_bytes()
    with pytest.raises(Exception):
        client._deserialize_model_optimized(payload)
    assert _GADGET_MARKER["executed"] is False


# ---------------------------------------------------------------------------
# (2) Server-side feature gates on _use_proxystore / _use_colab_connector /
#     _use_s3. Direct unit tests of the helper — full gRPC integration is
#     covered by the existing test suite on the happy path.
# ---------------------------------------------------------------------------


def _bare_servicer():
    from appfl.comm.grpc.grpc_server_communicator import GRPCServerCommunicator

    s = GRPCServerCommunicator.__new__(GRPCServerCommunicator)

    class _NullLogger:
        def warning(self, *a, **kw):
            pass

        def info(self, *a, **kw):
            pass

    s.logger = _NullLogger()
    return s


class _FakeContext:
    def __init__(self):
        self.code = None
        self.details = None

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details


def test_require_server_feature_allows_when_enabled():
    s = _bare_servicer()
    ctx = _FakeContext()
    # Must not raise, must not touch the context.
    s._require_server_feature("_use_proxystore", True, "client-1", ctx)
    assert ctx.code is None
    assert ctx.details is None


def test_require_server_feature_refuses_when_disabled():
    s = _bare_servicer()
    ctx = _FakeContext()
    with pytest.raises(ValueError, match="_use_proxystore"):
        s._require_server_feature("_use_proxystore", False, "client-1", ctx)
    assert ctx.code == grpc.StatusCode.FAILED_PRECONDITION
    assert "client-1" in ctx.details
    assert "_use_proxystore" in ctx.details


@pytest.mark.parametrize("flag", ["_use_proxystore", "_use_colab_connector", "_use_s3"])
def test_require_server_feature_message_names_flag(flag):
    """The refusal must name the offending flag so the client can debug."""
    s = _bare_servicer()
    ctx = _FakeContext()
    with pytest.raises(ValueError, match=flag):
        s._require_server_feature(flag, False, "alice", ctx)
    assert flag in ctx.details


# ---------------------------------------------------------------------------
# (3) validate_state_dict — structural schema enforcement.
# ---------------------------------------------------------------------------


from appfl.comm.utils.state_dict_validation import (  # noqa: E402
    InvalidStateDictPayload,
    validate_state_dict,
)


def _reference():
    return {
        "linear.weight": torch.zeros(3, 4),
        "linear.bias": torch.zeros(4),
    }


def test_validate_accepts_matching_state_dict():
    payload = {
        "linear.weight": torch.randn(3, 4),
        "linear.bias": torch.randn(4),
    }
    out = validate_state_dict(payload, reference=_reference())
    assert out is payload


def test_validate_accepts_subset_for_chunked_aggregation():
    """Chunked aggregation sends partial state dicts; that must be allowed
    by default (``allow_subset=True``)."""
    payload = {"linear.bias": torch.randn(4)}
    validate_state_dict(payload, reference=_reference())


def test_validate_rejects_subset_when_full_required():
    with pytest.raises(InvalidStateDictPayload, match="missing required keys"):
        validate_state_dict(
            {"linear.bias": torch.randn(4)},
            reference=_reference(),
            allow_subset=False,
        )


def test_validate_rejects_non_mapping():
    with pytest.raises(InvalidStateDictPayload, match="Mapping"):
        validate_state_dict([torch.zeros(3)], reference=_reference())


def test_validate_rejects_non_string_key():
    payload = {0: torch.zeros(4)}
    with pytest.raises(InvalidStateDictPayload, match="not a string"):
        validate_state_dict(payload, reference=_reference())


def test_validate_rejects_non_tensor_value():
    payload = {"linear.weight": [[1, 2, 3, 4]] * 3}
    with pytest.raises(InvalidStateDictPayload, match="not a torch.Tensor"):
        validate_state_dict(payload, reference=_reference())


def test_validate_rejects_unknown_key():
    payload = {
        "linear.weight": torch.randn(3, 4),
        "extra.injected": torch.randn(1),
    }
    with pytest.raises(InvalidStateDictPayload, match="unknown keys"):
        validate_state_dict(payload, reference=_reference())


def test_validate_rejects_wrong_shape():
    payload = {"linear.weight": torch.randn(99, 99)}
    with pytest.raises(InvalidStateDictPayload, match="shape"):
        validate_state_dict(payload, reference=_reference())


def test_validate_rejects_wrong_dtype():
    payload = {"linear.weight": torch.randn(3, 4).to(torch.float64)}
    with pytest.raises(InvalidStateDictPayload, match="dtype"):
        validate_state_dict(payload, reference=_reference())


def test_validate_without_reference_is_shape_agnostic():
    """When the server was started without a model architecture, the
    validator falls back to type checks only."""
    validate_state_dict({"any.key": torch.randn(7)})
    with pytest.raises(InvalidStateDictPayload):
        validate_state_dict({"any.key": "not a tensor"})


# Integration: _bytes_to_model now runs the validator and rejects an
# extra-key payload even if weights_only=True accepts it.


def test_bytes_to_model_rejects_extra_keys(tmp_path):
    """A torch-saved state dict with an extra parameter must be rejected
    against the server's reference, not silently aggregated."""
    from appfl.agent.server import ServerAgent

    agent = ServerAgent.__new__(ServerAgent)
    agent.enable_compression = False
    agent.optimize_memory = True

    # Plant a reference model that only has linear.weight.
    class _RefModel:
        def state_dict(self):
            return {"linear.weight": torch.zeros(3, 4)}

    agent.model = _RefModel()

    payload = {
        "linear.weight": torch.randn(3, 4),
        "evil.backdoor": torch.randn(99),
    }
    buf = io.BytesIO()
    torch.save(payload, buf)

    with pytest.raises(InvalidStateDictPayload):
        agent._bytes_to_model(buf.getvalue())


def test_bytes_to_model_rejects_wrong_shape_tensor():
    from appfl.agent.server import ServerAgent

    agent = ServerAgent.__new__(ServerAgent)
    agent.enable_compression = False
    agent.optimize_memory = True

    class _RefModel:
        def state_dict(self):
            return {"linear.weight": torch.zeros(3, 4)}

    agent.model = _RefModel()

    payload = {"linear.weight": torch.randn(99, 99)}
    buf = io.BytesIO()
    torch.save(payload, buf)

    with pytest.raises(InvalidStateDictPayload):
        agent._bytes_to_model(buf.getvalue())


def test_bytes_to_model_accepts_matching_payload():
    """Round-trip sanity: the happy path still works after adding the
    validator."""
    from appfl.agent.server import ServerAgent

    agent = ServerAgent.__new__(ServerAgent)
    agent.enable_compression = False
    agent.optimize_memory = True

    class _RefModel:
        def state_dict(self):
            return {
                "linear.weight": torch.zeros(3, 4),
                "linear.bias": torch.zeros(4),
            }

    agent.model = _RefModel()

    payload = {
        "linear.weight": torch.randn(3, 4),
        "linear.bias": torch.randn(4),
    }
    buf = io.BytesIO()
    torch.save(payload, buf)

    out = agent._bytes_to_model(buf.getvalue())
    assert set(out.keys()) == {"linear.weight", "linear.bias"}
