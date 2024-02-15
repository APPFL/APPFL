from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor
DONE: ServerStatus
ERROR: ServerStatus
RUN: ServerStatus

class ClientHeader(_message.Message):
    __slots__ = ["client_id"]
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    client_id: str
    def __init__(self, client_id: _Optional[str] = ...) -> None: ...

class ConfigurationRequest(_message.Message):
    __slots__ = ["header", "meta_data"]
    HEADER_FIELD_NUMBER: _ClassVar[int]
    META_DATA_FIELD_NUMBER: _ClassVar[int]
    header: ClientHeader
    meta_data: str
    def __init__(self, header: _Optional[_Union[ClientHeader, _Mapping]] = ..., meta_data: _Optional[str] = ...) -> None: ...

class ConfigurationResponse(_message.Message):
    __slots__ = ["configuration", "header"]
    CONFIGURATION_FIELD_NUMBER: _ClassVar[int]
    HEADER_FIELD_NUMBER: _ClassVar[int]
    configuration: str
    header: ServerHeader
    def __init__(self, header: _Optional[_Union[ServerHeader, _Mapping]] = ..., configuration: _Optional[str] = ...) -> None: ...

class CustomActionRequest(_message.Message):
    __slots__ = ["action", "header", "meta_data"]
    ACTION_FIELD_NUMBER: _ClassVar[int]
    HEADER_FIELD_NUMBER: _ClassVar[int]
    META_DATA_FIELD_NUMBER: _ClassVar[int]
    action: str
    header: ClientHeader
    meta_data: str
    def __init__(self, header: _Optional[_Union[ClientHeader, _Mapping]] = ..., action: _Optional[str] = ..., meta_data: _Optional[str] = ...) -> None: ...

class CustomActionResponse(_message.Message):
    __slots__ = ["header", "results"]
    HEADER_FIELD_NUMBER: _ClassVar[int]
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    header: ServerHeader
    results: str
    def __init__(self, header: _Optional[_Union[ServerHeader, _Mapping]] = ..., results: _Optional[str] = ...) -> None: ...

class DataBufferNew(_message.Message):
    __slots__ = ["data_bytes"]
    DATA_BYTES_FIELD_NUMBER: _ClassVar[int]
    data_bytes: bytes
    def __init__(self, data_bytes: _Optional[bytes] = ...) -> None: ...

class GetGlobalModelRequest(_message.Message):
    __slots__ = ["header", "meta_data"]
    HEADER_FIELD_NUMBER: _ClassVar[int]
    META_DATA_FIELD_NUMBER: _ClassVar[int]
    header: ClientHeader
    meta_data: str
    def __init__(self, header: _Optional[_Union[ClientHeader, _Mapping]] = ..., meta_data: _Optional[str] = ...) -> None: ...

class GetGlobalModelRespone(_message.Message):
    __slots__ = ["global_model", "header", "meta_data"]
    GLOBAL_MODEL_FIELD_NUMBER: _ClassVar[int]
    HEADER_FIELD_NUMBER: _ClassVar[int]
    META_DATA_FIELD_NUMBER: _ClassVar[int]
    global_model: bytes
    header: ServerHeader
    meta_data: str
    def __init__(self, header: _Optional[_Union[ServerHeader, _Mapping]] = ..., global_model: _Optional[bytes] = ..., meta_data: _Optional[str] = ...) -> None: ...

class ServerHeader(_message.Message):
    __slots__ = ["status"]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: ServerStatus
    def __init__(self, status: _Optional[_Union[ServerStatus, str]] = ...) -> None: ...

class UpdateGlobalModelRequest(_message.Message):
    __slots__ = ["header", "local_model", "meta_data"]
    HEADER_FIELD_NUMBER: _ClassVar[int]
    LOCAL_MODEL_FIELD_NUMBER: _ClassVar[int]
    META_DATA_FIELD_NUMBER: _ClassVar[int]
    header: ClientHeader
    local_model: bytes
    meta_data: str
    def __init__(self, header: _Optional[_Union[ClientHeader, _Mapping]] = ..., local_model: _Optional[bytes] = ..., meta_data: _Optional[str] = ...) -> None: ...

class UpdateGlobalModelResponse(_message.Message):
    __slots__ = ["global_model", "header", "meta_data"]
    GLOBAL_MODEL_FIELD_NUMBER: _ClassVar[int]
    HEADER_FIELD_NUMBER: _ClassVar[int]
    META_DATA_FIELD_NUMBER: _ClassVar[int]
    global_model: bytes
    header: ServerHeader
    meta_data: str
    def __init__(self, header: _Optional[_Union[ServerHeader, _Mapping]] = ..., global_model: _Optional[bytes] = ..., meta_data: _Optional[str] = ...) -> None: ...

class ServerStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
