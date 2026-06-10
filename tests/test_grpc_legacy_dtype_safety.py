"""Tests for src/appfl/comm/grpc_legacy/grpc_utils.py covering the fix for the
remote-code-execution vulnerability where a network-controlled tensor
``data_dtype`` string was resolved with ``eval()``.

A malicious gRPC peer (client -> server in ``grpc_server.py``, or server ->
client in ``grpc_client.py``) could previously set ``data_dtype`` to an
arbitrary Python expression that ``eval()`` would execute. ``parse_tensor_dtype``
must resolve only valid NumPy dtype names and never execute code.
"""

import numpy as np
import pytest

from appfl.comm.grpc_legacy.grpc_utils import (
    parse_tensor_dtype,
    construct_tensor_record,
)


@pytest.mark.parametrize(
    "wire_value, expected",
    [
        ("np.float32", np.dtype("float32")),
        ("np.float64", np.dtype("float64")),
        ("np.int8", np.dtype("int8")),
        ("np.int64", np.dtype("int64")),
        ("np.uint8", np.dtype("uint8")),
        ("float32", np.dtype("float32")),  # tolerated without the np. prefix
    ],
)
def test_valid_dtypes_resolve(wire_value, expected):
    assert parse_tensor_dtype(wire_value) == expected


def test_roundtrip_with_construct_tensor_record():
    """The dtype produced on the wire by construct_tensor_record must be
    parseable back to the original dtype."""
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    record = construct_tensor_record("w", arr)
    dtype = parse_tensor_dtype(record.data_dtype)
    restored = np.frombuffer(record.data_bytes, dtype=dtype).reshape(
        tuple(record.data_shape)
    )
    assert np.array_equal(arr, restored)
    assert restored.dtype == arr.dtype


@pytest.mark.parametrize(
    "payload",
    [
        '__import__("os").system("touch /tmp/appfl_pwned")',
        'exec("import os; os.system(\'id\')")',
        "np.float32 if open('/etc/passwd') else np.int8",
        "[].__class__.__base__",
        "lambda: 1",
        "np.dtype",  # callable, not a dtype spec
    ],
)
def test_malicious_payloads_raise_and_do_not_execute(payload, tmp_path):
    marker = "/tmp/appfl_pwned"
    import os

    if os.path.exists(marker):
        os.remove(marker)
    with pytest.raises(ValueError):
        parse_tensor_dtype(payload)
    assert not os.path.exists(marker), "payload executed — eval() regression"


def test_non_string_rejected():
    with pytest.raises(ValueError):
        parse_tensor_dtype(object())
