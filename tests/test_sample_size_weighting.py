"""Unit tests for client weights when client_weights_mode is "sample_size".

FedAvg used to weight a registered client by its share of the registered clients'
total sample size and a client with no registered size by 1 / n, so a round that
mixed the two had weights summing to more than 1 and overscaled the global step.
Such a round now falls back to equal weights, as in FedQueue. FedAsync and FedBuff
used to divide by zero when every registered sample size was 0; like FedAvg, they
now fall back to equal weights.
"""

import pytest
import torch
from omegaconf import OmegaConf

from appfl.algorithm.aggregator import (
    FedAdagradAggregator,
    FedAdamAggregator,
    FedAsyncAggregator,
    FedAvgAggregator,
    FedAvgMAggregator,
    FedBuffAggregator,
    FedYogiAggregator,
)


def _make(cls, mode, **configs):
    # OmegaConf rather than a plain dict: optimize_memory, K and server_* are read
    # as attributes.
    configs = OmegaConf.create({"client_weights_mode": mode, **configs})
    return cls(model=None, aggregator_configs=configs, logger=None)


def _upload(values):
    # Fresh tensors per upload: some aggregators update what they receive in place.
    return {cid: {"w": torch.tensor([float(v)])} for cid, v in values.items()}


def _fedavg_round(agg, chunked, values):
    kwargs = {}
    if chunked:
        # Streamed aggregation of the only chunk, which holds the only key.
        kwargs = {
            "_chunk_idx": {cid: 0 for cid in values},
            "_chunk_keys": {cid: ["w"] for cid in values},
        }
    return agg.aggregate(_upload(values), **kwargs)["w"].item()


# Every place FedAvg computes client weights: both branches of `compute_steps`,
# and `_compute_chunk_steps` for streamed aggregation.
_fedavg_paths = pytest.mark.parametrize(
    "configs, chunked",
    [
        ({"optimize_memory": True}, False),
        ({"optimize_memory": False}, False),
        ({}, True),
    ],
    ids=["optimize_memory", "unoptimized", "chunked"],
)


@_fedavg_paths
def test_fedavg_unregistered_client_does_not_overscale_step(configs, chunked):
    agg = _make(FedAvgAggregator, "sample_size", **configs)
    agg.set_client_sample_size("A", 100)  # B never registers.
    _fedavg_round(agg, chunked, {"A": 0, "B": 0})
    # Both clients move to 1, so any convex combination is 1; this used to be 1.5.
    assert _fedavg_round(agg, chunked, {"A": 1, "B": 1}) == pytest.approx(1.0)


@_fedavg_paths
def test_fedavg_unregistered_client_gives_equal_weights(configs, chunked):
    agg = _make(FedAvgAggregator, "sample_size", **configs)
    agg.set_client_sample_size("A", 100)
    agg.set_client_sample_size("B", 300)  # C never registers.
    _fedavg_round(agg, chunked, {"A": 0, "B": 0, "C": 0})
    # The mean of 3, 6 and 9. Weighting A and B by size within a 2/3 share would
    # give 6.5; the old weights, which summed to 4/3, gave 8.25.
    out = _fedavg_round(agg, chunked, {"A": 3, "B": 6, "C": 9})
    assert out == pytest.approx(6.0)


@_fedavg_paths
@pytest.mark.parametrize(
    "sizes, expected",
    [
        ({"A": 100, "B": 300}, 4.0),  # 0.25 * 1 + 0.75 * 5
        ({"A": 0, "B": 0}, 3.0),  # all sizes 0: equal weights
    ],
    ids=["by_size", "all_zero"],
)
def test_fedavg_all_clients_registered(configs, chunked, sizes, expected):
    agg = _make(FedAvgAggregator, "sample_size", **configs)
    for cid, size in sizes.items():
        agg.set_client_sample_size(cid, size)
    _fedavg_round(agg, chunked, {"A": 0, "B": 0})
    assert _fedavg_round(agg, chunked, {"A": 1, "B": 5}) == pytest.approx(expected)


# Server-optimizer settings required by the FedAvg subclasses.
_SERVER_OPT = {
    "server_learning_rate": 0.1,
    "server_adapt_param": 0.01,
    "server_momentum_param_1": 0.9,
    "server_momentum_param_2": 0.99,
}


@pytest.mark.parametrize("optimize_memory", [True, False])
@pytest.mark.parametrize(
    "cls",
    [FedAvgMAggregator, FedAdamAggregator, FedYogiAggregator, FedAdagradAggregator],
)
def test_server_optimizers_inherit_equal_weight_fallback(cls, optimize_memory):
    """The FedAvg subclasses build their step on `FedAvgAggregator.compute_steps`."""
    mixed = _make(cls, "sample_size", optimize_memory=optimize_memory, **_SERVER_OPT)
    mixed.set_client_sample_size("A", 100)
    mixed.set_client_sample_size("B", 300)  # C never registers.
    equal = _make(cls, "equal", optimize_memory=optimize_memory, **_SERVER_OPT)
    for values in (
        {"A": 0, "B": 0, "C": 0},
        {"A": 3, "B": 6, "C": 9},
        {"A": -2, "B": 1, "C": 4},
    ):
        assert torch.equal(
            mixed.aggregate(_upload(values))["w"],
            equal.aggregate(_upload(values))["w"],
        )


@pytest.mark.parametrize(
    "cls, configs",
    [
        (FedAsyncAggregator, {"num_clients": 2}),
        (FedBuffAggregator, {"num_clients": 2, "K": 2}),
    ],
    ids=["FedAsync", "FedBuff"],
)
@pytest.mark.parametrize(
    "sizes, weight_of_b",
    [
        ({"A": 100, "B": 300}, 0.75),  # B's share of the registered sample sizes
        ({"A": 0, "B": 0}, 0.5),  # all sizes 0: equal weight 1 / num_clients
    ],
    ids=["by_size", "all_zero"],
)
def test_async_all_clients_registered(cls, configs, sizes, weight_of_b):
    agg = _make(cls, "sample_size", **configs)
    for cid, size in sizes.items():
        agg.set_client_sample_size(cid, size)
    # With every size 0, this used to raise ZeroDivisionError.
    agg.aggregate("A", {"w": torch.zeros(1)})
    out = agg.aggregate("B", {"w": torch.ones(1)})["w"].item()
    # B's update is scaled by the default alpha of 0.9 times its weight.
    assert out == pytest.approx(0.9 * weight_of_b)
