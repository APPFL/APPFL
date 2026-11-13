"""Unit tests for the one-time warning aggregators log when client_weights_mode is
"sample_size" but a client being aggregated has no registered sample size.

Without it, a launcher that never calls ServerAgent.set_sample_size silently
trains with equal client weights.
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
from appfl.privacy.secure_aggregator import SecureAggregator


# --------------------------- helpers --------------------------- #
class _RecordingLogger:
    def __init__(self):
        self.warnings = []

    def info(self, msg):
        pass

    def warning(self, msg):
        self.warnings.append(msg)


def _update(model):
    # Fresh tensors per upload: some aggregators update what they receive in place.
    return {k: torch.randn_like(v) for k, v in model.state_dict().items()}


def _sync_round(agg, client_ids):
    agg.aggregate({cid: _update(agg.model) for cid in client_ids})


def _async_round(agg, client_ids):
    for cid in client_ids:
        # Record the download, as launchers do, so FedAsync's separate
        # "no recorded model download" warning stays quiet.
        agg.get_parameters(client_id=cid)
        agg.aggregate(cid, _update(agg.model))


# Server-optimizer settings required by the FedAvg subclasses.
_SERVER_OPT = {
    "server_learning_rate": 0.1,
    "server_adapt_param": 0.01,
    "server_momentum_param_1": 0.9,
    "server_momentum_param_2": 0.99,
}

# name -> (aggregator class, extra aggregator configs, how a round reaches it)
_CASES = {
    "FedAvg": (FedAvgAggregator, {}, _sync_round),
    "FedAvg-unoptimized": (FedAvgAggregator, {"optimize_memory": False}, _sync_round),
    "FedAvgM": (FedAvgMAggregator, _SERVER_OPT, _sync_round),
    "FedAdam": (FedAdamAggregator, _SERVER_OPT, _sync_round),
    "FedYogi": (FedYogiAggregator, _SERVER_OPT, _sync_round),
    "FedAdagrad": (FedAdagradAggregator, _SERVER_OPT, _sync_round),
    "FedAsync": (FedAsyncAggregator, {"num_clients": 2}, _async_round),
    "FedBuff": (FedBuffAggregator, {"num_clients": 2, "K": 2}, _async_round),
}


def _make(case, mode, logger):
    cls, extra_configs, run_round = _CASES[case]
    # OmegaConf rather than a plain dict: optimize_memory, K and server_* are
    # read as attributes.
    configs = OmegaConf.create({"client_weights_mode": mode, **extra_configs})
    agg = cls(model=torch.nn.Linear(2, 1), aggregator_configs=configs, logger=logger)
    return agg, run_round


# --------------------------- tests --------------------------- #
@pytest.mark.parametrize("case", list(_CASES))
def test_warns_once_naming_the_unregistered_client(case):
    logger = _RecordingLogger()
    agg, run_round = _make(case, "sample_size", logger)
    agg.set_client_sample_size("c0", 10)

    run_round(agg, ["c0", "c1"])
    assert len(logger.warnings) == 1
    msg = logger.warnings[0]
    assert "'c1'" in msg and "'c0'" not in msg
    assert "ServerAgent.set_sample_size" in msg
    assert "appfl.vsim.register_sample_sizes" in msg

    # Once per aggregator, not per client: a new unregistered client stays quiet.
    run_round(agg, ["c0", "c2"])
    assert len(logger.warnings) == 1


@pytest.mark.parametrize("case", list(_CASES))
def test_silent_when_all_sample_sizes_registered(case):
    logger = _RecordingLogger()
    agg, run_round = _make(case, "sample_size", logger)
    agg.set_client_sample_size("c0", 10)
    agg.set_client_sample_size("c1", 30)

    run_round(agg, ["c0", "c1"])
    run_round(agg, ["c0", "c1"])
    assert logger.warnings == []


@pytest.mark.parametrize("case", list(_CASES))
def test_silent_in_equal_mode(case):
    logger = _RecordingLogger()
    agg, run_round = _make(case, "equal", logger)

    run_round(agg, ["c0", "c1"])
    assert logger.warnings == []


def test_fedavg_chunked_aggregation_warns_once():
    logger = _RecordingLogger()
    agg, _ = _make("FedAvg", "sample_size", logger)
    agg.set_client_sample_size("c0", 10)
    client_ids = ["c0", "c1"]

    for _ in range(2):
        for chunk_idx, chunk_keys in enumerate([["weight"], ["bias"]]):
            update = {cid: _update(agg.model) for cid in client_ids}
            agg.aggregate(
                {cid: {k: update[cid][k] for k in chunk_keys} for cid in client_ids},
                _chunk_idx={cid: chunk_idx for cid in client_ids},
                _chunk_keys={cid: chunk_keys for cid in client_ids},
            )
    assert len(logger.warnings) == 1
    assert "'c1'" in logger.warnings[0]


def test_fedavg_secure_aggregation_does_not_warn():
    """The masked branch weights by each payload's num_examples, not registered sizes."""
    logger = _RecordingLogger()
    model = torch.nn.Linear(2, 1)
    configs = OmegaConf.create(
        {"client_weights_mode": "sample_size", "use_secure_agg": True}
    )
    agg = FedAvgAggregator(model=model, aggregator_configs=configs, logger=logger)

    def masked_payload(num_examples):
        flat, shapes = SecureAggregator.flatten_state_dict(_update(model))
        return {
            "type": "masked_update_flat",
            "flat": flat,
            "shapes": shapes,
            "num_examples": num_examples,
        }

    agg.aggregate({"c0": masked_payload(5), "c1": masked_payload(15)})
    assert logger.warnings == []


def test_no_logger_does_not_raise():
    agg, run_round = _make("FedAvg", "sample_size", logger=None)
    run_round(agg, ["c0", "c1"])


def test_long_client_lists_are_truncated():
    logger = _RecordingLogger()
    agg, run_round = _make("FedAvg", "sample_size", logger)

    # Nothing registered at all, so `client_sample_size` was never created.
    run_round(agg, [f"c{i}" for i in range(7)])
    assert len(logger.warnings) == 1
    msg = logger.warnings[0]
    assert "'c4'" in msg and "'c5'" not in msg
    assert "and 2 more" in msg
