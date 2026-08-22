import io
import torch
from .grpc_communicator_pb2 import DataBuffer


def proto_to_databuffer(proto, max_message_size=(2 * 1024 * 1024)):
    max_message_size = int(0.9 * max_message_size)
    data_bytes = proto.SerializeToString()
    data_bytes_size = len(data_bytes)
    message_size = (
        data_bytes_size if max_message_size > data_bytes_size else max_message_size
    )

    for i in range(0, data_bytes_size, message_size):
        chunk = data_bytes[i : i + message_size]
        msg = DataBuffer(data_bytes=chunk)
        yield msg


def serialize_model(model):
    """Serialize a model to a byte string."""
    buffer = io.BytesIO()
    torch.save(model, buffer)
    return buffer.getvalue()


def deserialize_model(model_bytes, *, weights_only: bool = True):
    """Deserialize a model from a byte string.

    By default, this calls :func:`torch.load` with ``weights_only=True`` so
    that arbitrary pickle payloads embedded in ``model_bytes`` cannot execute
    Python during load. This is the correct setting for the common case where
    the bytes hold a tensor state dict received from an untrusted peer.

    Callers that legitimately need to round-trip non-tensor Python objects
    (e.g. ``proxystore.proxy.Proxy`` references, Colab handles, or
    ``CloudStorageObject`` references) must pass ``weights_only=False``
    explicitly *and* ensure the operation is gated by a server-side opt-in to
    the corresponding feature — otherwise an attacker can put a pickle gadget
    in ``model_bytes`` and trigger code execution.
    """
    return torch.load(io.BytesIO(model_bytes), weights_only=weights_only)


def load_credential_from_file(filepath):
    with open(filepath, "rb") as f:
        return f.read()
