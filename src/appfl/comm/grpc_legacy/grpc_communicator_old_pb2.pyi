from collections.abc import Iterable as _Iterable
from collections.abc import Mapping as _Mapping
from typing import (
    ClassVar as _ClassVar,
)

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper

DESCRIPTOR: _descriptor.FileDescriptor
EMPTY: MessageStatus
INIT: Job
OK: MessageStatus
QUIT: Job
TRAIN: Job
WEIGHT: Job

class Acknowledgment(_message.Message):
    __slots__ = ["header", "status"]
    HEADER_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    header: Header
    status: MessageStatus
    def __init__(
        self,
        header: Header | _Mapping | None = ...,
        status: MessageStatus | str | None = ...,
    ) -> None: ...

class DataBufferV0(_message.Message):
    __slots__ = ["data_bytes", "size"]
    DATA_BYTES_FIELD_NUMBER: _ClassVar[int]
    SIZE_FIELD_NUMBER: _ClassVar[int]
    data_bytes: bytes
    size: int
    def __init__(
        self, size: int | None = ..., data_bytes: bytes | None = ...
    ) -> None: ...

class Header(_message.Message):
    __slots__ = ["client_id", "server_id"]
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    SERVER_ID_FIELD_NUMBER: _ClassVar[int]
    client_id: int
    server_id: int
    def __init__(
        self, server_id: int | None = ..., client_id: int | None = ...
    ) -> None: ...

class JobRequest(_message.Message):
    __slots__ = ["header", "job_done"]
    HEADER_FIELD_NUMBER: _ClassVar[int]
    JOB_DONE_FIELD_NUMBER: _ClassVar[int]
    header: Header
    job_done: Job
    def __init__(
        self,
        header: Header | _Mapping | None = ...,
        job_done: Job | str | None = ...,
    ) -> None: ...

class JobResponse(_message.Message):
    __slots__ = ["header", "job_todo", "round_number"]
    HEADER_FIELD_NUMBER: _ClassVar[int]
    JOB_TODO_FIELD_NUMBER: _ClassVar[int]
    ROUND_NUMBER_FIELD_NUMBER: _ClassVar[int]
    header: Header
    job_todo: Job
    round_number: int
    def __init__(
        self,
        header: Header | _Mapping | None = ...,
        round_number: int | None = ...,
        job_todo: Job | str | None = ...,
    ) -> None: ...

class LearningResults(_message.Message):
    __slots__ = ["dual", "header", "penalty", "primal", "round_number"]
    DUAL_FIELD_NUMBER: _ClassVar[int]
    HEADER_FIELD_NUMBER: _ClassVar[int]
    PENALTY_FIELD_NUMBER: _ClassVar[int]
    PRIMAL_FIELD_NUMBER: _ClassVar[int]
    ROUND_NUMBER_FIELD_NUMBER: _ClassVar[int]
    dual: _containers.RepeatedCompositeFieldContainer[TensorRecord]
    header: Header
    penalty: float
    primal: _containers.RepeatedCompositeFieldContainer[TensorRecord]
    round_number: int
    def __init__(
        self,
        header: Header | _Mapping | None = ...,
        round_number: int | None = ...,
        penalty: float | None = ...,
        primal: _Iterable[TensorRecord | _Mapping] | None = ...,
        dual: _Iterable[TensorRecord | _Mapping] | None = ...,
    ) -> None: ...

class TensorRecord(_message.Message):
    __slots__ = ["data_bytes", "data_dtype", "data_shape", "name"]
    DATA_BYTES_FIELD_NUMBER: _ClassVar[int]
    DATA_DTYPE_FIELD_NUMBER: _ClassVar[int]
    DATA_SHAPE_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    data_bytes: bytes
    data_dtype: str
    data_shape: _containers.RepeatedScalarFieldContainer[int]
    name: str
    def __init__(
        self,
        name: str | None = ...,
        data_shape: _Iterable[int] | None = ...,
        data_bytes: bytes | None = ...,
        data_dtype: str | None = ...,
    ) -> None: ...

class TensorRequest(_message.Message):
    __slots__ = ["header", "name", "round_number"]
    HEADER_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    ROUND_NUMBER_FIELD_NUMBER: _ClassVar[int]
    header: Header
    name: str
    round_number: int
    def __init__(
        self,
        header: Header | _Mapping | None = ...,
        name: str | None = ...,
        round_number: int | None = ...,
    ) -> None: ...

class WeightRequest(_message.Message):
    __slots__ = ["header", "size"]
    HEADER_FIELD_NUMBER: _ClassVar[int]
    SIZE_FIELD_NUMBER: _ClassVar[int]
    header: Header
    size: int
    def __init__(
        self,
        header: Header | _Mapping | None = ...,
        size: int | None = ...,
    ) -> None: ...

class WeightResponse(_message.Message):
    __slots__ = ["header", "weight"]
    HEADER_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_FIELD_NUMBER: _ClassVar[int]
    header: Header
    weight: float
    def __init__(
        self,
        header: Header | _Mapping | None = ...,
        weight: float | None = ...,
    ) -> None: ...

class Job(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class MessageStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
