"""Tests for `DFLNodeFedAvgAggregator`.

The `examples/dfl` FedAvg-equivalence run exercises exactly one path through this class:
uniform weights over a dict of neighbors, on a model with no integer buffers. Everything else
is covered here.
"""

import pytest
import torch
from omegaconf import OmegaConf

from appfl.algorithm.aggregator import DFLNodeFedAvgAggregator


@pytest.fixture
def aggregator():
    return DFLNodeFedAvgAggregator(
        model=None, aggregator_configs=OmegaConf.create({}), logger=None
    )


def value(state, name="w"):
    return state[name].item()


LOCAL = {"w": torch.tensor([0.0])}
PEERS = {"B": {"w": torch.tensor([10.0])}, "C": {"w": torch.tensor([20.0])}}


def test_uniform_average_over_the_closed_neighborhood(aggregator):
    """(0 + 10 + 20) / 3 -- the node's own model counts once, like a neighbor's."""
    assert value(aggregator.aggregate(LOCAL, PEERS, "A")) == pytest.approx(10.0)


def test_explicit_weights_are_applied_per_node(aggregator):
    weights = {"A": 0.5, "B": 0.25, "C": 0.25}
    result = aggregator.aggregate(LOCAL, PEERS, "A", mixing_weights=weights)
    assert value(result) == pytest.approx(0.5 * 0 + 0.25 * 10 + 0.25 * 20)


def test_weights_are_normalized_rather_than_trusted(aggregator):
    """Ratios are what matter, so a caller need not pre-normalize."""
    weights = {"A": 2.0, "B": 1.0, "C": 1.0}
    result = aggregator.aggregate(LOCAL, PEERS, "A", mixing_weights=weights)
    assert value(result) == pytest.approx(7.5)


def test_neighbor_models_may_be_a_list(aggregator):
    """No ids to key on, so weights cannot apply -- it falls back to a plain average."""
    assert value(aggregator.aggregate(LOCAL, list(PEERS.values()), "A")) == pytest.approx(10.0)


def test_an_isolated_node_keeps_its_own_model(aggregator):
    """A graph with no edges is the no-collaboration baseline, not an error."""
    assert value(aggregator.aggregate(LOCAL, {}, "A")) == pytest.approx(0.0)
    assert value(aggregator.aggregate(LOCAL, [], "A")) == pytest.approx(0.0)


def test_weights_summing_to_zero_are_refused(aggregator):
    with pytest.raises(ValueError, match="cannot be normalized"):
        aggregator.aggregate(
            LOCAL, PEERS, "A", mixing_weights={"A": 0.0, "B": 0.0, "C": 0.0}
        )


def test_integer_buffers_keep_their_dtype(aggregator):
    """`num_batches_tracked` is int64, and an averaged model must not turn it into a float.

    The example CNN is Conv/Linear only, so no run in `examples/` would reveal a mistake here.
    """
    local = {"n": torch.tensor([4], dtype=torch.long)}
    peers = {"B": {"n": torch.tensor([7], dtype=torch.long)}}
    result = aggregator.aggregate(local, peers, "A")["n"]
    assert result.dtype == torch.long
    assert result.item() == 5  # 5.5 truncated by the cast back to int


def test_every_parameter_is_mixed_not_just_the_first(aggregator):
    local = {"a": torch.zeros(2), "b": torch.ones(3)}
    peers = {"B": {"a": torch.full((2,), 4.0), "b": torch.full((3,), 3.0)}}
    result = aggregator.aggregate(local, peers, "A")
    assert torch.allclose(result["a"], torch.full((2,), 2.0))
    assert torch.allclose(result["b"], torch.full((3,), 2.0))


def test_the_aggregated_model_is_written_back_when_one_was_supplied():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent / "resources"))
    from tiny_model import TinyNet

    model = TinyNet()
    aggregator = DFLNodeFedAvgAggregator(
        model=model, aggregator_configs=OmegaConf.create({}), logger=None
    )
    local = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
    peers = {"B": {k: torch.ones_like(v) for k, v in model.state_dict().items()}}
    aggregator.aggregate(local, peers, "A")
    for tensor in model.state_dict().values():
        assert torch.allclose(tensor, torch.full_like(tensor, 0.5))


def test_get_parameters_needs_a_model(aggregator):
    with pytest.raises(ValueError, match="without a model"):
        aggregator.get_parameters()


def test_aggregate_does_not_mutate_its_inputs(aggregator):
    local = {"w": torch.tensor([2.0])}
    peers = {"B": {"w": torch.tensor([6.0])}}
    aggregator.aggregate(local, peers, "A")
    assert value(local) == pytest.approx(2.0)
    assert value(peers["B"]) == pytest.approx(6.0)


def test_a_node_takes_the_remainder_when_no_weight_names_it(aggregator):
    """Weights given for the neighbors alone: the node's own share is what is left.

    The alternative -- guessing which unclaimed key belongs to this node -- picks arbitrarily
    when more than one is unclaimed, which is a wrong answer rather than an error.
    """
    weights = {"B": 0.25, "C": 0.25}
    result = aggregator.aggregate(LOCAL, PEERS, "A", mixing_weights=weights)
    assert value(result) == pytest.approx(0.5 * 0 + 0.25 * 10 + 0.25 * 20)


def test_an_unrelated_key_is_not_mistaken_for_this_node(aggregator):
    """A stale or misspelled id in the dict must not become this node's weight."""
    weights = {"B": 0.25, "C": 0.25, "Stale": 0.9}
    result = aggregator.aggregate(LOCAL, PEERS, "A", mixing_weights=weights)
    assert value(result) == pytest.approx(7.5)  # as if "Stale" were not there
