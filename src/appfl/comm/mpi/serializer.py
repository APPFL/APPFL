import io
import torch
import pickle
from dataclasses import fields
from typing import Dict, OrderedDict, Union
from .config import MPITaskRequest, MPITaskResponse


def byte_to_request(byte_obj: bytes) -> MPITaskRequest:
    """Convert a byte object to a MPITaskRequest object."""
    request = MPITaskRequest()
    obj = pickle.loads(byte_obj)
    request.payload = obj["payload"]
    request.meta_data = obj["meta_data"]
    return request


def request_to_byte(request: MPITaskRequest) -> bytes:
    """Convert a MPITaskRequest object to a byte object."""
    request_dict = {
        field.name: getattr(request, field.name) for field in fields(request)
    }
    return pickle.dumps(request_dict)


def byte_to_response(byte_obj: bytes) -> MPITaskResponse:
    """Convert a byte object to a MPITaskResponse object."""
    response = MPITaskResponse()
    obj = pickle.loads(byte_obj)
    response.status = obj["status"]
    response.payload = obj["payload"]
    response.meta_data = obj["meta_data"]
    return response


def response_to_byte(response: MPITaskResponse) -> bytes:
    """Convert a MPITaskResponse object to a byte object."""
    response_dict = {
        field.name: getattr(response, field.name) for field in fields(response)
    }
    return pickle.dumps(response_dict)


def model_to_byte(model: Union[Dict, OrderedDict]) -> bytes:
    """Serialize a model to a byte string."""
    buffer = io.BytesIO()
    torch.save(model, buffer)
    return buffer.getvalue()


def byte_to_model(byte_obj: bytes) -> Union[Dict, OrderedDict]:
    """Deserialize a byte string to a model."""
    return torch.load(io.BytesIO(byte_obj))
