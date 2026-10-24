"""Unit tests for the vsim flat-buffer state-dict codec (appfl.vsim.codec)."""

import pytest

torch = pytest.importorskip("torch")

from appfl.vsim.codec import (  # noqa: E402
    make_layout,
    pack_state,
    same_layout,
    schema_nbytes,
    unpack_state,
)


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


# ------------------------------ layouts ------------------------------ #
def _model_state():
    return {
        "conv.weight": torch.randn(4, 3),
        "bn.weight": torch.randn(4),
        "bn.running_mean": torch.randn(4),
        "bn.num_batches_tracked": torch.tensor(5, dtype=torch.int64),
        "fc.bias": torch.randn(7),
    }


def test_make_layout_follows_the_given_order_and_keeps_the_schema_order():
    state = _model_state()
    order = ["conv.weight", "bn.weight", "fc.bias", "bn.running_mean"]
    order.append("bn.num_batches_tracked")
    layout = make_layout(state, order)
    assert [entry[0] for entry in layout] == list(state)
    starts = {name: start for name, _, _, start, _ in layout}
    assert [starts[name] for name in order] == sorted(starts.values())
    assert all(start % 64 == 0 for start in starts.values())
    schema, buf = pack_state(state, layout=layout)
    assert schema == layout and buf.numel() == schema_nbytes(layout)
    out = unpack_state(schema, buf)
    assert list(out) == list(state)
    assert all(torch.equal(out[name], state[name]) for name in state)


def test_make_layout_rejects_an_order_that_is_not_a_permutation():
    state = _model_state()
    with pytest.raises(ValueError, match="exactly once"):
        make_layout(state, list(state)[:-1])
    with pytest.raises(ValueError, match="exactly once"):
        make_layout(state, list(state) + ["conv.weight"])


def test_pack_state_reuses_a_layout_in_any_name_order():
    state = _model_state()
    layout = make_layout(state, list(reversed(state)))
    shuffled = {name: state[name] for name in ["fc.bias", *list(state)[:-1]]}
    schema, buf = pack_state(shuffled, layout=layout)
    assert [entry[0] for entry in schema] == list(shuffled)  # the state's order
    assert same_layout(schema, layout) and schema != layout
    out = unpack_state(schema, buf)
    assert list(out) == list(shuffled)
    assert all(torch.equal(out[name], state[name]) for name in state)


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.pop("fc.bias"),
        lambda s: s.update(extra=torch.zeros(2)),
        lambda s: s.update({"fc.bias": torch.zeros(8)}),
        lambda s: s.update({"fc.bias": torch.zeros(7, dtype=torch.float64)}),
    ],
    ids=["missing", "extra", "shape", "dtype"],
)
def test_pack_state_lays_out_a_different_state_in_its_own_order(change):
    state = _model_state()
    layout = make_layout(state, list(reversed(state)))
    change(state)
    schema, buf = pack_state(state, layout=layout)
    assert schema == make_layout(state) and not same_layout(schema, layout)
    out = unpack_state(schema, buf)
    assert all(torch.equal(out[name], state[name]) for name in state)


def test_same_layout_compares_placement_not_order():
    state = _model_state()
    layout = make_layout(state)
    assert same_layout(layout, list(reversed(layout)))
    assert not same_layout(layout, make_layout(state, list(reversed(state))))
    assert not same_layout(layout, layout[:-1])
