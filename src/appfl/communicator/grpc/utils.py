import io
import torch
from .grpc_communicator_new_pb2 import DataBufferNew

def proto_to_databuffer_new(proto, max_message_size=(2 * 1024 * 1024)):
    max_message_size = int(0.9 * max_message_size)
    data_bytes = proto.SerializeToString()
    data_bytes_size = len(data_bytes)
    message_size = (
        data_bytes_size if max_message_size > data_bytes_size else max_message_size
    )

    for i in range(0, data_bytes_size, message_size):
        chunk = data_bytes[i : i + message_size]
        msg = DataBufferNew(data_bytes=chunk)
        yield msg


def serialize_model(model):
    """Serialize a model to a byte string."""
    buffer = io.BytesIO()
    torch.save(model, buffer)
    return buffer.getvalue()