import copy

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import TensorDataset
from appfl.algorithm.scheduler import SyncScheduler
from appfl.algorithm.trainer import TCSTrainer
from appfl.algorithm.aggregator import TCSAggregator


class MeanLoss(torch.nn.Module):
    def forward(self, prediction, target):
        return prediction.mean()


def trainer(gradients=(1.0, 0.0, 0.0), **config):
    x = torch.as_tensor(gradients, dtype=torch.float64)
    if x.ndim == 1:
        x = x[:, None]
    model = torch.nn.Linear(x.shape[1], 1, bias=False).double()
    torch.nn.init.zeros_(model.weight)
    cfg = dict(
        mode="step",
        num_base_steps=1,
        num_local_steps=10,
        kappa=0.0,
        optim="SGD",
        optim_args={"lr": 1.0},
        train_batch_size=1,
        train_data_shuffle=False,
    )
    cfg.update(config)
    return TCSTrainer(
        model=model,
        loss_fn=MeanLoss(),
        train_dataset=TensorDataset(x, torch.zeros(len(x))),
        train_configs=OmegaConf.create(cfg),
    )


def test_global_round_zero_is_base_only():
    t = trainer(num_base_steps=2)
    t.train(round=0)
    assert t.metadata["num_steps"] == 2
    assert t.metadata["stop_reason"] == "base"


def test_first_participation_after_round_zero_is_adaptive():
    t = trainer()
    t.train(round=7)
    assert t.metadata["num_steps"] == 2
    assert t.metadata["psi"] == 0
    assert t.metadata["stop_reason"] == "trajectory"
    # The stopping step is kept, not undone.
    assert t.model.weight.item() == -1


def test_zero_reference_stops_at_base():
    t = trainer((0.0, 9.0))
    t.train(round=1)
    assert t.metadata["num_steps"] == 1
    assert t.metadata["norm_ratio"] is None


def test_turn_stops_even_without_magnitude_decay():
    t = trainer(((1.0, 0.0), (-1.0, 2.0)))
    t.train(round=1)
    assert t.metadata["num_steps"] == 2
    assert t.metadata["norm_ratio"] == 1
    assert t.metadata["cosine"] == 0


@pytest.mark.parametrize("epochs", [2, 5, 10])
def test_epoch_cap_allows_multiple_epochs(epochs):
    t = trainer((1.0,) * 5, mode="epoch", num_local_epochs=epochs, kappa=0.5)
    t.train(round=1)
    assert t.base == 5
    assert t.metadata["num_steps"] == 5 * epochs
    assert t.metadata["local_epochs"] == epochs


@pytest.mark.parametrize("kappa,expected", [(0.9, 1.2), (0.6, 1.8)])
def test_fractional_epoch_after_one_epoch_reference(kappa, expected):
    t = trainer(
        (1.0, 1.0, 1.0, 1.0, 6.0), mode="epoch", num_local_epochs=10, kappa=kappa
    )
    t.train(round=1)
    assert t.metadata["stop_reason"] == "trajectory"
    assert t.metadata["local_epochs"] == expected


def test_epoch_round_zero_and_partial_final_batch():
    t = trainer((1.0,) * 5, mode="epoch", num_local_epochs=5, train_batch_size=2)
    t.train(round=0)
    assert t.metadata["num_steps"] == 3
    assert t.metadata["examples_seen"] == 5
    assert t.metadata["local_epochs"] == 1


def test_residual_cap_and_no_incomplete_base():
    t = trainer((1.0,), num_base_steps=2)
    t.train(round=1, local_steps=4)
    assert t.metadata["num_steps"] == 4
    with pytest.raises(ValueError, match="base"):
        t.train(round=1, local_steps=1)


@pytest.mark.parametrize(
    "config",
    [
        {"kappa": 1},
        {"kappa": -0.1},
        {"kappa": float("nan")},
        {"num_base_steps": 0},
        {"num_local_steps": 0},
        {"num_local_steps": True},
        {"num_base_steps": 5, "num_local_steps": 2},
        {"mode": "bad"},
        {"mode": "epoch", "num_local_epochs": 0},
        {"optim": "Adam"},
        {"optim_args": {"lr": 0.1, "momentum": 0.9}},
        {"optim_args": {"lr": 0.1, "weight_decay": 0.1}},
        {"optim_args": {"lr": float("inf")}},
        {"use_dp": True},
        {"clip_grad": True},
    ],
)
def test_invalid_configuration_fails(config):
    with pytest.raises(ValueError):
        trainer(**config)


def make_aggregator():
    model = torch.nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        model.weight.fill_(2.0)
    return TCSAggregator(model)


def updates():
    return {
        "a": {"weight": torch.tensor([[4.0]], dtype=torch.float64)},
        "b": {"weight": torch.tensor([[10.0]], dtype=torch.float64)},
    }


def metadata():
    return dict(
        num_steps={"a": 1, "b": 4},
        num_samples={"a": 1, "b": 3},
        round={"a": 0, "b": 0},
        learning_rate={"a": 0.1, "b": 0.1},
    )


def test_work_normalization_hand_calculated():
    a = make_aggregator()
    result, meta = a.aggregate(updates(), **metadata())
    # Hbar=3.25, weighted delta/steps=2, theta=2+3.25*2=8.5.
    assert result["weight"].item() == 8.5
    assert meta == {"round": 1}
    result["weight"].zero_()
    assert a.get_parameters()[0]["weight"].item() == 8.5


def test_equal_steps_recover_sample_weighted_fedavg():
    a = make_aggregator()
    m = metadata()
    m["num_steps"] = {"a": 3, "b": 3}
    result, _ = a.aggregate(updates(), **m)
    assert result["weight"].item() == 8.5


def test_unequal_steps_differ_from_fedavg():
    a = make_aggregator()
    m = metadata()
    m["num_samples"] = {"a": 1, "b": 1}
    models = updates()
    models["b"]["weight"].fill_(6.0)
    result, _ = a.aggregate(models, **m)
    assert result["weight"].item() == 5.75
    assert result["weight"].item() != 5.0


@pytest.mark.parametrize(
    "key,value",
    [
        ("num_steps", 0),
        ("num_steps", 1.5),
        ("num_steps", True),
        ("num_samples", 0),
        ("round", 1),
        ("round", None),
        ("learning_rate", 0.2),
        ("learning_rate", float("nan")),
    ],
)
def test_invalid_metadata_is_atomic(key, value):
    a = make_aggregator()
    m = metadata()
    m[key]["a"] = value
    with pytest.raises(ValueError):
        a.aggregate(updates(), **m)
    assert a.round == 0
    assert a.get_parameters()[0]["weight"].item() == 2


def test_registered_samples_and_tensor_validation():
    a = make_aggregator()
    a.set_client_sample_size("a", 2)
    with pytest.raises(ValueError, match="registered"):
        a.aggregate(updates(), **metadata())
    a.set_client_sample_size("a", 1)
    bad = updates()
    bad["a"]["weight"].fill_(float("nan"))
    with pytest.raises(ValueError, match="tensor"):
        a.aggregate(bad, **metadata())


def test_sync_scheduler_preserves_metadata_and_resolves_both_clients():
    a = make_aggregator()
    scheduler = SyncScheduler(OmegaConf.create({"num_clients": 2}), a, None)
    m = metadata()
    f1 = scheduler.schedule("a", updates()["a"], **{k: v["a"] for k, v in m.items()})
    assert not f1.done()
    f2 = scheduler.schedule("b", updates()["b"], **{k: v["b"] for k, v in m.items()})
    assert f1.done() and f2.done()
    assert f1.result()[1] == f2.result()[1] == {"round": 1}
    assert scheduler.get_num_global_epochs() == 1


def test_no_extra_forward_for_stopping_and_snapshot_reset():
    t = trainer((1.0,), num_local_steps=4)
    calls = []
    t.model.register_forward_hook(lambda *args: calls.append(1))
    t.train(round=1)
    assert len(calls) == t.metadata["num_steps"] == 4
    state = copy.deepcopy(t.model.state_dict())
    state["weight"].fill_(12.0)
    t.load_parameters(state)
    assert t.get_parameters()["weight"].item() == 12


def test_quadratic_matches_closed_form_and_keeps_stopping_iterate():
    t = trainer((1.0,), kappa=0.5, optim_args={"lr": 0.1})
    with torch.no_grad():
        t.model.weight.fill_(1.0)
    t.loss_fn = lambda prediction, target: 0.5 * prediction.square().mean()
    t.train(round=1)
    # theta_h=.9**h, d_h=(1-.9**h)/(.1*h); first d_h <= .75 is h=7.
    assert t.metadata["num_steps"] == 7
    assert t.model.weight.item() == pytest.approx(0.9**7, abs=1e-12)


def test_common_learning_rate_override_changes_actual_sgd():
    t = trainer((1.0,), num_local_steps=2)
    t.train(round=1, learning_rate=0.25)
    assert t.model.weight.item() == -0.5
    assert t.metadata["learning_rate"] == 0.25


def test_buffer_policy_and_atomic_rejection():
    model = torch.nn.Linear(1, 1, bias=False).double()
    model.register_buffer("average", torch.tensor(0.0, dtype=torch.float64))
    model.register_buffer("counter", torch.tensor(0))
    a = TCSAggregator(model)
    models = {cid: copy.deepcopy(model.state_dict()) for cid in ["a", "b"]}
    models["a"]["average"].fill_(4.0)
    models["b"]["average"].fill_(8.0)
    models["b"]["counter"].fill_(1)
    with pytest.raises(ValueError, match="buffer"):
        a.aggregate(models, **metadata())
    assert a.round == 0
    models["a"]["counter"].fill_(1)
    result, _ = a.aggregate(models, **metadata())
    assert result["average"].item() == 7.0
    assert result["counter"].item() == 1


@pytest.mark.parametrize("mutation", ["missing", "shape", "dtype"])
def test_malformed_models_rejected(mutation):
    a = make_aggregator()
    models = updates()
    if mutation == "missing":
        models["a"] = {}
    elif mutation == "shape":
        models["a"]["weight"] = torch.zeros(2, dtype=torch.float64)
    else:
        models["a"]["weight"] = models["a"]["weight"].float()
    with pytest.raises(ValueError):
        a.aggregate(models, **metadata())
    assert a.round == 0


def test_initial_model_is_required_and_empty_aggregation_rejected():
    with pytest.raises(ValueError, match="initial"):
        TCSAggregator()
    with pytest.raises(ValueError, match="no client"):
        make_aggregator().aggregate({})


def test_serial_tuple_response_sets_global_round_for_late_participant():
    t = trainer()
    t.load_parameters((t.model.state_dict(), {"round": 4}))
    t.train()
    assert t.metadata["round"] == 4
    assert t.metadata["num_steps"] == 2
