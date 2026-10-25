import numpy as np
from .grpc_communicator_old_pb2 import DataBufferV0
from .grpc_communicator_old_pb2 import TensorRecord


def parse_tensor_dtype(data_dtype: str) -> np.dtype:
    """
    Safely resolve the ``data_dtype`` field of a `TensorRecord` into a NumPy
    dtype.

    The wire format produced by :func:`construct_tensor_record` prefixes the
    dtype name with ``"np."`` (e.g. ``"np.float32"``). Historically this field
    was resolved with ``eval()``, which let a malicious peer execute arbitrary
    Python by sending a crafted ``data_dtype`` string. This helper instead
    strips the optional ``"np."`` prefix and resolves the name through
    ``numpy.dtype``, which only accepts valid dtype specifiers and never
    executes code.
    """
    if not isinstance(data_dtype, str):
        raise ValueError(f"data_dtype must be a string, got {type(data_dtype)!r}")
    name = data_dtype[3:] if data_dtype.startswith("np.") else data_dtype
    try:
        return np.dtype(name)
    except TypeError as e:
        raise ValueError(f"Unsupported or invalid tensor dtype: {data_dtype!r}") from e


def construct_tensor_record(name, nparray):
    return TensorRecord(
        name=name,
        data_shape=list(nparray.shape),
        data_bytes=nparray.tobytes(order="C"),
        data_dtype="np." + str(nparray.dtype),
    )


def proto_to_databuffer(proto, max_message_size=(2 * 1024 * 1024)):
    max_message_size = max_message_size - 16  # 16 bytes for the message size field
    data_bytes = proto.SerializeToString()
    data_bytes_size = len(data_bytes)
    message_size = (
        data_bytes_size if max_message_size > data_bytes_size else max_message_size
    )

    for i in range(0, data_bytes_size, message_size):
        chunk = data_bytes[i : i + message_size]
        msg = DataBufferV0(size=message_size, data_bytes=chunk)
        yield msg
