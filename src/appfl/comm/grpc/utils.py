import io
import torch
from typing import Any, Dict, Optional
from .grpc_communicator_pb2 import DataBuffer

MAX_RECEIVE_MESSAGE_BYTES: int = 256 * 1024 * 1024

def response_chunk_size(
    own_max_message_size: int, meta_data: Optional[Dict[str, Any]] = None
) -> int:
    """Chunk a response to the smaller of what this server sends and what the caller accepts.

    Every request carries the caller's own ``max_message_size`` in its ``meta_data``. Chunking
    a reply larger than that is refused at the caller's channel, so the two are reconciled
    here. Absent a declaration, the server's own size stands.
    """
    if not meta_data:
        return own_max_message_size
    return min(
        own_max_message_size,
        int(meta_data.get("max_message_size", own_max_message_size)),
    )


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


def deserialize_model(model_bytes):
    """Deserialize a model from a byte string."""
    return torch.load(io.BytesIO(model_bytes))


def load_credential_from_file(filepath):
    with open(filepath, "rb") as f:
        return f.read()
