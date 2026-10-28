"""Unit tests for the vsim flat-buffer state-dict codec (appfl.vsim.codec)."""

import pytest

torch = pytest.importorskip("torch")

from appfl.vsim.codec import pack_state, schema_nbytes, unpack_state  # noqa: E402


def _state():
    base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    return {
        "w": torch.randn(5, 3),
        "half": torch.randn(7).half(),
        "bf16": torch.randn(3, 3).to(torch.bfloat16),
        "double": torch.randn(2, dtype=torch.float64),
        "steps": torch.tensor(17, dtype=torch.int64),  # 0-dim, like num_batches_tracked
        "mask": torch.tensor([True, False, True]),
        "bytes": torch.arange(5, dtype=torch.uint8),
        "empty": torch.empty(0, 4),
        "transposed": base.t(),  # non-contiguous
        "param": torch.nn.Parameter(torch.ones(2, 2)),  # requires grad
    }


def test_round_trip_preserves_values_dtypes_shapes_and_order():
    state = _state()
    schema, buf = pack_state(state)
    out = unpack_state(schema, buf)
    assert list(out) == list(state)
    for name, tensor in state.items():
        assert out[name].dtype == tensor.dtype, name
        assert out[name].shape == tensor.shape, name
        assert torch.equal(out[name], tensor.detach()), name


def test_offsets_are_aligned_and_buffer_size_matches_schema():
    schema, buf = pack_state(_state())
    assert all(offset % 64 == 0 for _, _, _, offset, _ in schema)
    assert buf.dtype == torch.uint8 and buf.dim() == 1
    assert buf.numel() == schema_nbytes(schema)


def test_unpacked_tensors_are_views_of_the_buffer():
    schema, buf = pack_state({"w": torch.zeros(4)})
    out = unpack_state(schema, buf)
    out["w"].fill_(1.0)
    assert torch.equal(unpack_state(schema, buf)["w"], torch.ones(4))


def test_pack_copies_its_input():
    state = {"w": torch.zeros(3)}
    schema, buf = pack_state(state)
    state["w"].fill_(5.0)
    assert torch.equal(unpack_state(schema, buf)["w"], torch.zeros(3))


def test_empty_state_round_trips():
    schema, buf = pack_state({})
    assert schema == [] and buf.numel() == 0 and schema_nbytes(schema) == 0
    assert unpack_state(schema, buf) == {}


def test_non_tensor_values_are_rejected():
    with pytest.raises(TypeError, match="dense tensors only"):
        pack_state({"w": torch.zeros(2), "lr": 0.1})


def test_compressed_payloads_are_rejected():
    # A client with enable_compression returns its update as bytes.
    with pytest.raises(TypeError, match="Disable model compression"):
        pack_state(b"compressed update")


def test_schema_is_plain_python():
    """The schema is pickled into MPI headers, so it must hold no torch objects."""
    import pickle

    schema, _ = pack_state(_state())
    assert pickle.loads(pickle.dumps(schema)) == schema
    for name, dtype_name, shape, offset, nbytes in schema:
        assert isinstance(dtype_name, str) and isinstance(shape, tuple)
        assert all(isinstance(x, int) for x in (*shape, offset, nbytes))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_tensors_are_packed_from_host_copies():
    schema, buf = pack_state({"w": torch.ones(3, device="cuda")})
    assert torch.equal(unpack_state(schema, buf)["w"], torch.ones(3))
