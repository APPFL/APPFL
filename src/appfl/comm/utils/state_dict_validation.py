"""Strict schema validation for incoming model state dicts.

In APPFL's federated-learning protocol a client is supposed to return an
updated version of the *server's* model — same architecture, same set of
parameter names, same per-parameter shape and dtype, just different tensor
values. A client is **not** supposed to send arbitrary Python objects, or a
state dict for a different architecture, or extra keys.

This module provides :func:`validate_state_dict`, which makes that contract
explicit at the receiving end. It is a structural defence on top of
``torch.load(weights_only=True)``: even a payload that survives the weights-only
unpickler must still satisfy the federation's known schema, or it is rejected
before anything else touches it.

The reference schema is derived from ``server_agent.model.state_dict()`` and
passed in by the caller; if no reference is supplied (e.g. the server was
configured without a model architecture, which happens for some compression-only
deployments), the validator falls back to a shape-agnostic check that the
payload is at least a mapping of strings to tensors.
"""

from __future__ import annotations

from typing import Mapping

import torch


class InvalidStateDictPayload(ValueError):
    """Raised when an incoming model payload does not match the federation's
    expected state-dict schema. Catching code should treat this as a protocol
    error (the client sent something that cannot possibly be a legitimate
    update), not as a transient failure."""


def validate_state_dict(
    payload: object,
    *,
    reference: Mapping[str, torch.Tensor] | None = None,
    allow_subset: bool = True,
) -> Mapping[str, torch.Tensor]:
    """Validate that ``payload`` is a state dict compatible with ``reference``.

    :param payload: The object that was deserialised from the wire (or
        extracted from proxystore / S3 / colab transport). Anything other
        than a ``Mapping[str, torch.Tensor]`` is rejected.
    :param reference: Optional mapping of expected parameter names to tensors
        whose ``shape`` and ``dtype`` define the legal schema. When supplied,
        every key in ``payload`` must be present in ``reference`` and every
        tensor's shape and dtype must match.
    :param allow_subset: When ``True`` (the default) the payload may carry
        only a subset of the reference keys, which is the case for chunked
        aggregation. Set to ``False`` to require the payload to cover the
        full key set.
    :return: ``payload`` (unchanged) when validation succeeds. Returning the
        value lets callers chain ``model = validate_state_dict(model, ...)``.
    :raises InvalidStateDictPayload: When the payload fails any check.
    """
    if not isinstance(payload, Mapping):
        raise InvalidStateDictPayload(
            f"expected a Mapping[str, torch.Tensor] state dict, got "
            f"{type(payload).__name__}"
        )

    for key, value in payload.items():
        if not isinstance(key, str):
            raise InvalidStateDictPayload(f"state dict key {key!r} is not a string")
        if not isinstance(value, torch.Tensor):
            raise InvalidStateDictPayload(
                f"state dict value for key {key!r} is not a torch.Tensor "
                f"(got {type(value).__name__})"
            )

    if reference is None:
        return payload

    ref_keys = set(reference.keys())
    payload_keys = set(payload.keys())

    unknown = payload_keys - ref_keys
    if unknown:
        raise InvalidStateDictPayload(
            f"state dict contains unknown keys not present in the server's "
            f"model: {sorted(unknown)[:5]}" + ("..." if len(unknown) > 5 else "")
        )

    if not allow_subset:
        missing = ref_keys - payload_keys
        if missing:
            raise InvalidStateDictPayload(
                f"state dict is missing required keys: {sorted(missing)[:5]}"
                + ("..." if len(missing) > 5 else "")
            )

    for key, value in payload.items():
        ref_tensor = reference[key]
        if tuple(value.shape) != tuple(ref_tensor.shape):
            raise InvalidStateDictPayload(
                f"state dict tensor {key!r} has shape {tuple(value.shape)}, "
                f"expected {tuple(ref_tensor.shape)}"
            )
        if value.dtype != ref_tensor.dtype:
            raise InvalidStateDictPayload(
                f"state dict tensor {key!r} has dtype {value.dtype}, "
                f"expected {ref_tensor.dtype}"
            )

    return payload
