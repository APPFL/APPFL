"""Flat-buffer codec for model state dicts.

Packs a ``{name: tensor}`` mapping into one contiguous ``uint8`` tensor plus a small
picklable schema, and unpacks it again as zero-copy tensor views. The distributed
simulator moves the buffer with raw-buffer MPI sends, so a model is never pickled or
passed through ``torch.save``.

Every tensor starts at a 64-byte-aligned offset, which keeps the reinterpreting
``view(dtype)`` used by :func:`unpack_state` valid for every dtype, and the schema
preserves insertion order, which aggregators rely on. Tensors are laid out in
insertion order unless the caller plans another layout (:func:`make_layout`), so
offsets need not increase along the schema.
"""

from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple

_ALIGN = 64

# (name, dtype name, shape, byte offset, byte count)
SchemaEntry = Tuple[str, str, Tuple[int, ...], int, int]


class PackedState(NamedTuple):
    """A state dict packed by :func:`pack_state`: its schema and its byte buffer."""

    schema: List[SchemaEntry]
    buffer: Any


def _aligned(offset: int) -> int:
    """Round ``offset`` up to the next multiple of the alignment."""
    return (offset + _ALIGN - 1) // _ALIGN * _ALIGN


def _describe(name: str, value) -> Tuple[str, Tuple[int, ...], int]:
    """Return (dtype name, shape, byte count) of a dense tensor, or raise."""
    import torch

    if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
        raise TypeError(
            f"Cannot pack {name!r}: the distributed simulator moves dense tensors "
            f"only, got {type(value).__name__}. Disable model compression and "
            f"return a plain state dict from the trainer."
        )
    dtype_name = str(value.dtype).replace("torch.", "")
    return dtype_name, tuple(value.shape), value.numel() * value.element_size()


def make_layout(state, order: Optional[Iterable[str]] = None) -> List[SchemaEntry]:
    """
    Plan where each tensor of ``state`` goes in a packed buffer, without packing it.

    :param state: Mapping of names to dense tensors.
    :param order: Every name of ``state`` once, in the order their bytes are laid
        out. ``None`` lays them out in ``state``'s own order.
    :return: Schema listing the names in ``state``'s order, for the ``layout`` of
        :func:`pack_state`.
    :raises ValueError: If ``order`` is not a permutation of the names in ``state``.
    """
    described = {name: _describe(name, value) for name, value in state.items()}
    names = list(described) if order is None else list(order)
    if len(names) != len(described) or set(names) != set(described):
        raise ValueError("order must list every name of the state exactly once")
    starts = {}
    offset = 0
    for name in names:
        offset = _aligned(offset)
        starts[name] = offset
        offset += described[name][2]
    return [
        (name, dtype_name, shape, starts[name], nbytes)
        for name, (dtype_name, shape, nbytes) in described.items()
    ]


def _follow(layout: List[SchemaEntry], state) -> Optional[List[SchemaEntry]]:
    """Re-list ``layout`` in the order of ``state`` if both hold the same tensors."""
    if len(layout) != len(state):
        return None
    by_name = {entry[0]: entry for entry in layout}
    schema = []
    for name, value in state.items():
        entry = by_name.get(name)
        dtype_name, shape, _ = _describe(name, value)
        if entry is None or entry[1] != dtype_name or entry[2] != shape:
            return None
        schema.append(entry)
    return schema


def pack_state(state, layout: Optional[List[SchemaEntry]] = None):
    """
    Pack a state dict into one contiguous byte buffer.

    :param state: Mapping of names to tensors (any device; copied to CPU).
    :param layout: Optional schema whose byte layout to reuse. If ``state`` holds
        exactly its names, dtypes and shapes, in any order, every tensor goes to its
        offset there; otherwise ``state`` is laid out in its own order.
    :return: Tuple of (schema, buffer), where buffer is a CPU ``torch.uint8`` tensor
        and the schema lists the names in the order of ``state``.
    :raises TypeError: If a value is not a dense tensor.
    """
    import torch

    schema = _follow(layout, state) if layout is not None else None
    if schema is None:
        schema = make_layout(state)

    # Zero-filled so alignment padding is deterministic rather than stale memory.
    buffer = torch.zeros(schema_nbytes(schema), dtype=torch.uint8)
    for name, _, _, start, nbytes in schema:
        if nbytes:
            tensor = state[name].detach()
            if tensor.device.type != "cpu":
                tensor = tensor.cpu()
            tensor = tensor.contiguous()
            buffer[start : start + nbytes].copy_(tensor.reshape(-1).view(torch.uint8))
    return schema, buffer


def unpack_state(schema: List[SchemaEntry], buffer) -> Dict:
    """
    Unpack a buffer produced by :func:`pack_state` into zero-copy tensor views.

    The views share memory with ``buffer``, so the caller must not reuse the buffer
    while any of the returned tensors is alive. Aggregators may keep or modify them
    in place.

    :param schema: Schema returned by :func:`pack_state`.
    :param buffer: CPU ``torch.uint8`` tensor holding the packed bytes.
    :return: Dict of tensors in the original insertion order.
    """
    import torch

    state = {}
    for name, dtype_name, shape, start, nbytes in schema:
        dtype = getattr(torch, dtype_name)
        if nbytes == 0:
            state[name] = torch.empty(shape, dtype=dtype)
        else:
            state[name] = buffer[start : start + nbytes].view(dtype).view(shape)
    return state


def schema_nbytes(schema: List[SchemaEntry]) -> int:
    """Return the buffer size, in bytes, that ``schema`` describes."""
    return max((_aligned(start + nbytes) for *_, start, nbytes in schema), default=0)


def same_layout(a: List[SchemaEntry], b: List[SchemaEntry]) -> bool:
    """Whether two schemas put the same tensors at the same offsets, in any order."""
    return a == b or (len(a) == len(b) and sorted(a) == sorted(b))
