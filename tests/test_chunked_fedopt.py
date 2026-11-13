"""Full/chunk parity and independent optimizer checks using synthetic tensors."""

import copy

import pytest
import torch
from omegaconf import OmegaConf

from appfl.algorithm.aggregator import (
    FedAdagradAggregator,
    FedAdamAggregator,
    FedAvgAggregator,
    FedAvgMAggregator,
    FedYogiAggregator,
)
from appfl.misc.memory_utils import split_state_dict_by_size


AGGREGATORS = [
    FedAvgAggregator,
    FedAvgMAggregator,
    FedAdamAggregator,
    FedYogiAggregator,
    FedAdagradAggregator,
]


class BufferedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.25, -0.5]))
        self.bias = torch.nn.Parameter(torch.tensor([0.1, 0.2, 0.3]))
        self.register_buffer("running_mean", torch.zeros(2))
        self.register_buffer("counter", torch.tensor(1, dtype=torch.int64))


def config(optimize_memory, weighting):
    return OmegaConf.create(
        {
            "optimize_memory": optimize_memory,
            "client_weights_mode": weighting,
            "server_learning_rate": 0.4,
            "server_adapt_param": 0.1,
            "server_momentum_param_1": 0.5,
            "server_momentum_param_2": 0.75,
        }
    )


def make_aggregator(kind, optimize_memory, weighting, with_model=True):
    aggregator = kind(
        model=BufferedModel() if with_model else None,
        aggregator_configs=config(optimize_memory, weighting),
    )
    aggregator.set_client_sample_size("a", 1)
    aggregator.set_client_sample_size("b", 3)
    return aggregator


def local_models(round_idx):
    initial = BufferedModel().state_dict()
    return {
        client: {
            name: value + (round_idx + 1) * offset for name, value in initial.items()
        }
        for client, offset in [("a", 2), ("b", 6)]
    }


def aggregate_chunks(aggregator, models, chunk_size):
    chunks = list(split_state_dict_by_size(models["a"], chunk_size))
    result = {}
    for chunk_idx, _, keys in chunks:
        chunk_models = {
            client: {key: state[key] for key in keys}
            for client, state in models.items()
        }
        result.update(
            aggregator.aggregate(
                chunk_models,
                _chunk_idx={client: chunk_idx for client in models},
                _total_chunks={client: len(chunks) for client in models},
                _chunk_keys={client: keys for client in models},
            )
        )
    return result


def assert_state_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])


def assert_optimizer_equal(actual, expected):
    for name in ["m_vector", "v_vector"]:
        if hasattr(expected, name):
            assert_state_equal(getattr(actual, name), getattr(expected, name))


@pytest.mark.parametrize("kind", AGGREGATORS)
@pytest.mark.parametrize("optimize_memory", [False, True])
@pytest.mark.parametrize("weighting", ["equal", "sample_size"])
@pytest.mark.parametrize("with_model", [False, True])
@pytest.mark.parametrize("chunk_size", [8, 20])
def test_chunked_optimizer_matches_full_over_rounds(
    kind, optimize_memory, weighting, with_model, chunk_size
):
    full = make_aggregator(kind, optimize_memory, weighting, with_model)
    chunked = make_aggregator(kind, optimize_memory, weighting, with_model)
    for round_idx in range(3):
        models = local_models(round_idx)
        expected = full.aggregate(copy.deepcopy(models))
        actual = aggregate_chunks(chunked, copy.deepcopy(models), chunk_size)
        assert_state_equal(actual, expected)
        assert_optimizer_equal(chunked, full)


@pytest.mark.parametrize("kind", AGGREGATORS[1:])
@pytest.mark.parametrize("optimize_memory", [False, True])
@pytest.mark.parametrize("weighting", ["equal", "sample_size"])
def test_chunked_optimizer_matches_independent_equations(
    kind, optimize_memory, weighting
):
    aggregator = make_aggregator(kind, optimize_memory, weighting)
    expected = {
        name: value.clone() for name, value in BufferedModel().named_parameters()
    }
    momentum = {name: torch.zeros_like(value) for name, value in expected.items()}
    variance = {
        name: torch.full_like(value, 0.1**2) for name, value in expected.items()
    }
    weights = (
        {"a": 0.25, "b": 0.75} if weighting == "sample_size" else {"a": 0.5, "b": 0.5}
    )
    for round_idx in range(3):
        models = local_models(round_idx)
        for name, value in expected.items():
            delta = sum(
                weights[client] * (state[name] - value)
                for client, state in models.items()
            )
            if kind is FedAvgMAggregator:
                momentum[name] = 0.5 * momentum[name] + delta
                step = momentum[name]
            else:
                momentum[name] = 0.5 * momentum[name] + 0.5 * delta
                if kind is FedAdamAggregator:
                    variance[name] = 0.75 * variance[name] + 0.25 * delta.square()
                elif kind is FedYogiAggregator:
                    variance[name] -= (
                        0.25
                        * delta.square()
                        * torch.sign(variance[name] - delta.square())
                    )
                else:
                    variance[name] += delta.square()
                step = 0.4 * momentum[name] / (variance[name].sqrt() + 0.1)
            expected[name] = value + step
        actual = aggregate_chunks(aggregator, models, 8)
        assert_state_equal({name: actual[name] for name in expected}, expected)
        if kind is FedAvgMAggregator:
            assert_state_equal(aggregator.v_vector, momentum)
        else:
            assert_state_equal(aggregator.m_vector, momentum)
            assert_state_equal(aggregator.v_vector, variance)
        # Model buffers are averaged independently of parameter weights/optimizer.
        torch.testing.assert_close(
            actual["running_mean"],
            (models["a"]["running_mean"] + models["b"]["running_mean"]) / 2,
        )


@pytest.mark.parametrize("kind", AGGREGATORS[1:])
@pytest.mark.parametrize("optimize_memory", [False, True])
def test_full_and_different_chunk_boundaries_preserve_optimizer_state(
    kind, optimize_memory
):
    full = make_aggregator(kind, optimize_memory, "sample_size")
    mixed = make_aggregator(kind, optimize_memory, "sample_size")
    for round_idx, chunk_size in enumerate([None, 8, 20, None]):
        models = local_models(round_idx)
        expected = full.aggregate(copy.deepcopy(models))
        actual = (
            mixed.aggregate(copy.deepcopy(models))
            if chunk_size is None
            else aggregate_chunks(mixed, copy.deepcopy(models), chunk_size)
        )
        assert_state_equal(actual, expected)
        assert_optimizer_equal(mixed, full)


@pytest.mark.parametrize("optimize_memory", [False, True])
def test_secure_aggregation_remains_incompatible_with_chunks(optimize_memory):
    aggregator = make_aggregator(FedAdamAggregator, optimize_memory, "equal")
    aggregator.aggregator_configs.use_secure_agg = True
    with pytest.raises(AssertionError, match="does not support chunked"):
        aggregate_chunks(aggregator, local_models(0), 8)
