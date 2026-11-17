"""Tests for the validation of ``client_weights_mode`` in the aggregators.

The aggregators only implement the ``"equal"`` and ``"sample_size"`` weighting
modes. Any other value used to fall through to equal weights without notice, so
a typo such as ``"sample"`` silently ignored the clients' sample sizes. The
aggregators now reject such values when they are constructed.
"""

import pytest
from omegaconf import OmegaConf

from appfl.algorithm.aggregator import (
    FedAsyncAggregator,
    FedAvgAggregator,
    FedBuffAggregator,
)

# FedBuff reads its buffer size `K` in __init__, so its config needs one.
with_each_aggregator = pytest.mark.parametrize(
    "aggregator_cls, extra_configs",
    [
        (FedAvgAggregator, {}),
        (FedAsyncAggregator, {}),
        (FedBuffAggregator, {"K": 2}),
    ],
    ids=["FedAvg", "FedAsync", "FedBuff"],
)


def _build(aggregator_cls, extra_configs, **configs):
    return aggregator_cls(
        aggregator_configs=OmegaConf.create({**extra_configs, **configs})
    )


@with_each_aggregator
@pytest.mark.parametrize("mode", ["equal", "sample_size"])
def test_valid_client_weights_mode_accepted(aggregator_cls, extra_configs, mode):
    aggregator = _build(aggregator_cls, extra_configs, client_weights_mode=mode)
    assert aggregator.client_weights_mode == mode


@with_each_aggregator
def test_client_weights_mode_defaults_to_equal(aggregator_cls, extra_configs):
    assert _build(aggregator_cls, extra_configs).client_weights_mode == "equal"


@with_each_aggregator
# "num_examples" is a value of the separate `secure_agg_client_weights_mode`.
@pytest.mark.parametrize("mode", ["sample", "num_examples", "Sample_Size"])
def test_invalid_client_weights_mode_rejected(aggregator_cls, extra_configs, mode):
    with pytest.raises(ValueError) as excinfo:
        _build(aggregator_cls, extra_configs, client_weights_mode=mode)
    message = str(excinfo.value)
    assert repr(mode) in message
    assert "'equal'" in message and "'sample_size'" in message
