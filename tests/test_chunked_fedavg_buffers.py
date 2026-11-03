"""Chunked FedAvg preserves the full-model policy for parameters and buffers."""

import copy

import pytest
import torch
from omegaconf import OmegaConf

from appfl.algorithm.aggregator import FedAvgAggregator
from appfl.misc.memory_utils import split_state_dict_by_size


def _aggregator(model, optimize_memory, weighting):
    aggregator = FedAvgAggregator(
        model=copy.deepcopy(model),
        aggregator_configs=OmegaConf.create(
            {
                "optimize_memory": optimize_memory,
                "client_weights_mode": weighting,
            }
        ),
    )
    aggregator.set_client_sample_size("a", 1)
    aggregator.set_client_sample_size("b", 3)
    return aggregator


def _aggregate_chunks(aggregator, models, chunk_size):
    chunks = {
        client: split_state_dict_by_size(model, chunk_size)
        for client, model in copy.deepcopy(models).items()
    }
    total = len(chunks["a"])
    assert total > 1
    result = {}
    for i in range(total):
        result.update(
            aggregator.aggregate(
                {client: parts[i][1] for client, parts in chunks.items()},
                _chunk_idx={client: i for client in chunks},
                _total_chunks={client: total for client in chunks},
                _chunk_keys={client: parts[i][2] for client, parts in chunks.items()},
            )
        )
    return result


@pytest.mark.parametrize("optimize_memory", [False, True])
@pytest.mark.parametrize("weighting", ["equal", "sample_size"])
@pytest.mark.parametrize("chunk_size", [16, 32])
def test_chunked_batchnorm_preserves_full_model_buffers(
    optimize_memory, weighting, chunk_size
):
    model = torch.nn.BatchNorm1d(2, dtype=torch.float64)
    full = _aggregator(model, optimize_memory, weighting)
    chunked = _aggregator(model, optimize_memory, weighting)
    models = {}
    for client, parameter, mean, variance, batches in [
        ("a", 1.0, 10.0, 4.0, 2),
        ("b", 3.0, 30.0, 8.0, 4),
    ]:
        state = copy.deepcopy(model.state_dict())
        state["weight"].fill_(parameter)
        state["bias"].fill_(parameter)
        state["running_mean"].fill_(mean)
        state["running_var"].fill_(variance)
        state["num_batches_tracked"].fill_(batches)
        models[client] = state

    expected = full.aggregate(copy.deepcopy(models))
    actual = _aggregate_chunks(chunked, models, chunk_size)
    # Full-model aggregation averages non-parameter buffers equally. This is
    # a parity test for that existing policy, not a pooled-variance estimator.
    torch.testing.assert_close(
        expected["running_mean"], torch.full((2,), 20.0, dtype=torch.float64)
    )
    torch.testing.assert_close(
        expected["running_var"], torch.full((2,), 6.0, dtype=torch.float64)
    )
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name])
    assert actual["num_batches_tracked"].dtype == torch.int64
    assert actual["num_batches_tracked"].item() == 3

    # Verify the partially updated model itself, not just the returned chunks.
    inputs = torch.tensor([[15.0, 30.0]], dtype=torch.float64)
    full.model.eval()
    chunked.model.eval()
    with torch.no_grad():
        torch.testing.assert_close(chunked.model(inputs), full.model(inputs))


@pytest.mark.parametrize("optimize_memory", [False, True])
@pytest.mark.parametrize("chunk_size", [16, 32])
def test_model_free_tensors_still_use_sample_size_weights(optimize_memory, chunk_size):
    full = _aggregator(None, optimize_memory, "sample_size")
    chunked = _aggregator(None, optimize_memory, "sample_size")
    models = {
        client: {
            "w": torch.full((3,), value, dtype=torch.float64),
            "b": torch.full((2,), value, dtype=torch.float64),
        }
        for client, value in [("a", 1.0), ("b", 3.0)]
    }
    expected = full.aggregate(copy.deepcopy(models))
    actual = _aggregate_chunks(chunked, models, chunk_size)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name])
        torch.testing.assert_close(actual[name], torch.full_like(actual[name], 2.5))
