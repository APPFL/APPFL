from collections.abc import Mapping as _Mapping
from typing import (
    ClassVar as _ClassVar,
)

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper

DESCRIPTOR: _descriptor.FileDescriptor

class ServerStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RUN: _ClassVar[ServerStatus]
    DONE: _ClassVar[ServerStatus]
    ERROR: _ClassVar[ServerStatus]

RUN: ServerStatus
DONE: ServerStatus
ERROR: ServerStatus

class DataBuffer(_message.Message):
    __slots__ = ("data_bytes",)
    DATA_BYTES_FIELD_NUMBER: _ClassVar[int]
    data_bytes: bytes
    def __init__(self, data_bytes: bytes | None = ...) -> None: ...

class ClientHeader(_message.Message):
    __slots__ = ("client_id",)
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    client_id: str
    def __init__(self, client_id: str | None = ...) -> None: ...

class ServerHeader(_message.Message):
    __slots__ = ("status",)
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: ServerStatus
    def __init__(self, status: ServerStatus | str | None = ...) -> None: ...

class ConfigurationRequest(_message.Message):
    __slots__ = ("header", "meta_data")
    HEADER_FIELD_NUMBER: _ClassVar[int]
    META_DATA_FIELD_NUMBER: _ClassVar[int]
    header: ClientHeader
    meta_data: str
    def __init__(
        self,
        header: ClientHeader | _Mapping | None = ...,
        meta_data: str | None = ...,
    ) -> None: ...

class ConfigurationResponse(_message.Message):
    __slots__ = ("configuration", "header")
    HEADER_FIELD_NUMBER: _ClassVar[int]
    CONFIGURATION_FIELD_NUMBER: _ClassVar[int]
    header: ServerHeader
    configuration: str
    def __init__(
        self,
        header: ServerHeader | _Mapping | None = ...,
        configuration: str | None = ...,
    ) -> None: ...

class GetGlobalModelRequest(_message.Message):
    __slots__ = ("header", "meta_data")
    HEADER_FIELD_NUMBER: _ClassVar[int]
    META_DATA_FIELD_NUMBER: _ClassVar[int]
    header: ClientHeader
    meta_data: str
    def __init__(
        self,
        header: ClientHeader | _Mapping | None = ...,
        meta_data: str | None = ...,
    ) -> None: ...

class GetGlobalModelRespone(_message.Message):
    __slots__ = ("global_model", "header", "meta_data")
    HEADER_FIELD_NUMBER: _ClassVar[int]
    GLOBAL_MODEL_FIELD_NUMBER: _ClassVar[int]
    META_DATA_FIELD_NUMBER: _ClassVar[int]
    header: ServerHeader
    global_model: bytes
    meta_data: str
    def __init__(
        self,
        header: ServerHeader | _Mapping | None = ...,
        global_model: bytes | None = ...,
        meta_data: str | None = ...,
    ) -> None: ...

class UpdateGlobalModelRequest(_message.Message):
    __slots__ = ("header", "local_model", "meta_data")
    HEADER_FIELD_NUMBER: _ClassVar[int]
    LOCAL_MODEL_FIELD_NUMBER: _ClassVar[int]
    META_DATA_FIELD_NUMBER: _ClassVar[int]
    header: ClientHeader
    local_model: bytes
    meta_data: str
    def __init__(
        self,
        header: ClientHeader | _Mapping | None = ...,
        local_model: bytes | None = ...,
        meta_data: str | None = ...,
    ) -> None: ...

class UpdateGlobalModelResponse(_message.Message):
    __slots__ = ("global_model", "header", "meta_data")
    HEADER_FIELD_NUMBER: _ClassVar[int]
    GLOBAL_MODEL_FIELD_NUMBER: _ClassVar[int]
    META_DATA_FIELD_NUMBER: _ClassVar[int]
    header: ServerHeader
    global_model: bytes
    meta_data: str
    def __init__(
        self,
        header: ServerHeader | _Mapping | None = ...,
        global_model: bytes | None = ...,
        meta_data: str | None = ...,
    ) -> None: ...

class CustomActionRequest(_message.Message):
    __slots__ = ("action", "header", "meta_data")
    HEADER_FIELD_NUMBER: _ClassVar[int]
    ACTION_FIELD_NUMBER: _ClassVar[int]
    META_DATA_FIELD_NUMBER: _ClassVar[int]
    header: ClientHeader
    action: str
    meta_data: str
    def __init__(
        self,
        header: ClientHeader | _Mapping | None = ...,
        action: str | None = ...,
        meta_data: str | None = ...,
    ) -> None: ...

class CustomActionResponse(_message.Message):
    __slots__ = ("header", "results")
    HEADER_FIELD_NUMBER: _ClassVar[int]
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    header: ServerHeader
    results: str
    def __init__(
        self,
        header: ServerHeader | _Mapping | None = ...,
        results: str | None = ...,
    ) -> None: ...
