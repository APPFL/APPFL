"""Flat-buffer codec for model state dicts.

Packs a ``{name: tensor}`` mapping into one contiguous ``uint8`` tensor plus a small
picklable schema, and unpacks it again as zero-copy tensor views. The distributed
simulator moves the buffer with raw-buffer MPI sends, so a model is never pickled or
passed through ``torch.save``.

Every tensor starts at a 64-byte-aligned offset, which keeps the reinterpreting
``view(dtype)`` used by :func:`unpack_state` valid for every dtype, and the schema
preserves insertion order, which aggregators rely on.
"""

from typing import Any, Dict, List, Tuple

_ALIGN = 64

# (name, dtype name, shape, byte offset, byte count)
SchemaEntry = Tuple[str, str, Tuple[int, ...], int, int]


def _aligned(offset: int) -> int:
    """Round ``offset`` up to the next multiple of the alignment."""
    return (offset + _ALIGN - 1) // _ALIGN * _ALIGN


def pack_state(state) -> Tuple[List[SchemaEntry], Any]:
    """
    Pack a state dict into one contiguous byte buffer.

    :param state: Mapping of names to tensors (any device; copied to CPU).
    :return: Tuple of (schema, buffer), where buffer is a CPU ``torch.uint8`` tensor.
    :raises TypeError: If a value is not a dense tensor.
    """
    import torch

    schema: List[SchemaEntry] = []
    tensors = []
    offset = 0
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
            raise TypeError(
                f"Cannot pack {name!r}: the distributed simulator moves dense tensors "
                f"only, got {type(value).__name__}. Disable model compression and "
                f"return a plain state dict from the trainer."
            )
        tensor = value.detach()
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        tensor = tensor.contiguous()
        nbytes = tensor.numel() * tensor.element_size()
        offset = _aligned(offset)
        dtype_name = str(tensor.dtype).replace("torch.", "")
        schema.append((name, dtype_name, tuple(tensor.shape), offset, nbytes))
        tensors.append(tensor)
        offset += nbytes

    # Zero-filled so alignment padding is deterministic rather than stale memory.
    buffer = torch.zeros(_aligned(offset), dtype=torch.uint8)
    for (_, _, _, start, nbytes), tensor in zip(schema, tensors):
        if nbytes:
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
    if not schema:
        return 0
    _, _, _, start, nbytes = schema[-1]
    return _aligned(start + nbytes)
