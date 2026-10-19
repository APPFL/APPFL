"""Tests for SharedAgentPool and LightClient (no MPI needed).

Light clients train every virtual client on one shared ClientAgent. Driven by the
in-process engine, they must reproduce full ClientAgents exactly: the history, the
global model, and the round counter each update reports.
"""

import gc
import logging
import os
import types

import pytest

torch = pytest.importorskip("torch")

from omegaconf import OmegaConf  # noqa: E402

from appfl.vsim import (  # noqa: E402
    AsyncSimDriver,
    ClientProfile,
    LightClient,
    SharedAgentPool,
    SyncSimDriver,
    load_client_datasets,
)

RESOURCES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")
IDS = [f"Client{i + 1}" for i in range(4)]


def _logger():
    lg = logging.getLogger("vsim_light_client_test")
    lg.handlers.clear()
    lg.addHandler(logging.NullHandler())
    lg.setLevel(logging.CRITICAL)
    return lg


def _server(name, tmp_dir, epochs, validation):
    from appfl.agent import ServerAgent

    cfg = OmegaConf.load(os.path.join(RESOURCES, "configs", name))
    train = cfg.client_configs.train_configs
    train.num_local_steps = 3
    train.train_batch_size = 16
    train.do_validation = validation
    train.do_pre_validation = validation
    train.loss_fn_path = os.path.join(RESOURCES, "loss", "celoss.py")
    train.metric_path = os.path.join(RESOURCES, "metric", "acc.py")
    cfg.client_configs.model_configs.model_path = os.path.join(
        RESOURCES, "model", "cnn.py"
    )
    cfg.client_configs.model_configs.model_kwargs.num_pixel = 16
    cfg.server_configs.num_clients = len(IDS)
    cfg.server_configs.num_global_epochs = epochs
    cfg.server_configs.logging_output_dirname = tmp_dir
    return ServerAgent(server_agent_config=cfg)


def _client_config(cid, device="cpu"):
    return OmegaConf.create(
        {
            "client_id": cid,
            "train_configs": {"device": device, "logging_output_filename": ""},
            "data_configs": {
                "dataset_path": os.path.join(
                    RESOURCES, "dataset", "synthetic_dataset.py"
                ),
                "dataset_name": "get_synthetic",
                "dataset_kwargs": {
                    "num_clients": len(IDS),
                    "client_id": IDS.index(cid),
                },
            },
        }
    )


def _full_client(cid, context, device):
    from appfl.agent import ClientAgent

    client = ClientAgent(client_agent_config=_client_config(cid, device))
    client.load_config(OmegaConf.create(context))
    return client


def _data(cid, context, device):
    return load_client_datasets(_client_config(cid, device))


def _run(kind, tmp_dir, shared, validation):
    """Run one in-process simulation; return its history, model and rounds."""
    profiles = {cid: ClientProfile(compute_factor=1.0 + i) for i, cid in enumerate(IDS)}
    if kind == "fedasync":
        server = _server("server_fedasync.yaml", tmp_dir, 12, validation)
    else:
        server = _server("server_fedavg.yaml", tmp_dir, 3, validation)
    context = OmegaConf.to_container(server.get_client_configs(), resolve=True)
    if shared:
        pool = SharedAgentPool(_full_client, _data)
        clients = [pool.client(cid, context, "cpu") for cid in IDS]
    else:
        pool = None
        clients = [_full_client(cid, context, "cpu") for cid in IDS]
    rounds = []
    update = server.global_update

    def recording(client_id, local_model, blocking=False, **kw):
        rounds.append((client_id, kw.get("round"), kw.get("val_accuracy")))
        return update(
            client_id=client_id, local_model=local_model, blocking=blocking, **kw
        )

    server.global_update = recording
    common = dict(logger=_logger(), seed=3, base_step_time=0.01)
    if kind == "fedasync":
        driver = AsyncSimDriver(server, clients, profiles, max_in_flight=3, **common)
    else:
        # Over-selection: 3 dispatched, the first 2 kept, so the in-process engine
        # also trains clients the barrier then discards.
        driver = SyncSimDriver(
            server,
            clients,
            profiles,
            participants_per_round=3,
            mode="count",
            min_responses=2,
            target_rounds=3,
            **common,
        )
    gc.collect()
    gc.freeze()
    driver.run()
    gc.unfreeze()
    return driver.history, server.get_parameters(serial_run=True), rounds, pool


@pytest.mark.parametrize("validation", [False, True])
@pytest.mark.parametrize("kind", ["fedasync", "fedavg"])
def test_shared_clients_match_full_clients(tmp_path, kind, validation):
    torch.set_num_threads(1)
    history, model, rounds, _ = _run(kind, str(tmp_path), False, validation)
    shared_history, shared_model, shared_rounds, pool = _run(
        kind, str(tmp_path), True, validation
    )
    assert shared_history == history
    assert shared_rounds == rounds
    assert list(shared_model) == list(model)
    for name in model:
        assert torch.equal(shared_model[name], model[name]), name
    # One engine served all four clients, each with its own data.
    assert pool.stats["data_loads"] == len(IDS)
    assert pool.stats["binds"] > len(IDS)
    if validation and kind == "fedasync":
        assert all(acc is not None for _, _, acc in rounds)


def test_shared_clients_keep_no_model_state(tmp_path):
    torch.set_num_threads(1)
    server = _server("server_fedasync.yaml", str(tmp_path), 4, False)
    context = OmegaConf.to_container(server.get_client_configs(), resolve=True)
    pool = SharedAgentPool(_full_client, _data)
    clients = [pool.client(cid, context, "cpu") for cid in IDS]
    params = server.get_parameters(serial_run=True)
    for client in clients:
        client.load_parameters(params)
        client.train()
        model, meta = client.get_parameters()
        assert set(model) == set(params)
        assert meta["round"] == 1 and client.round == 1
        # The engine hands the local model over instead of keeping a copy.
        assert "model_state" not in vars(pool.agent.trainer)
    assert not hasattr(LightClient(pool, IDS[0]), "__dict__")
    assert {c.get_sample_size() for c in clients} == {48}
    assert [c.expected_local_steps() for c in clients] == [3] * len(IDS)
    clients[0].set_round(7)
    clients[0].load_parameters(params)
    clients[0].train()
    assert clients[0].get_parameters()[1]["round"] == 8


def test_a_new_context_rebuilds_the_engine(tmp_path):
    server = _server("server_fedasync.yaml", str(tmp_path), 4, False)
    context = OmegaConf.to_container(server.get_client_configs(), resolve=True)
    pool = SharedAgentPool(_full_client, _data)
    pool.client(IDS[0], context, "cpu")
    first = pool.agent
    pool.client(IDS[1], context, "cpu")
    assert pool.agent is first
    changed = dict(context, train_configs=dict(context["train_configs"]))
    changed["train_configs"]["num_local_steps"] = 5
    assert pool.client(IDS[1], changed, "cpu").expected_local_steps() == 5
    assert pool.agent is not first


# ------------------------- refusals ------------------------- #
class _Trainer:
    def __init__(self, **train_configs):
        self.train_configs = OmegaConf.create(train_configs)


class _Model:
    def state_dict(self):
        return {"weight": 0, "bias": 0, "running_mean": 0}


def _fake_agent(trainer):
    def make(cid, context, device):
        return types.SimpleNamespace(
            trainer=trainer,
            model=_Model(),
            train_dataset=[0] * 8,
            val_dataset=None,
            load_parameters=lambda params: None,
        )

    return make


def _no_data(cid, context, device):
    raise AssertionError("not needed")


def test_trainers_with_unknown_state_are_refused():
    pool = SharedAgentPool(_fake_agent(_Trainer(mode="step")), _no_data)
    with pytest.raises(TypeError, match="keep no per-client state"):
        pool.client("C1")


def test_opacus_privacy_accounting_is_refused():
    pool = SharedAgentPool(
        _fake_agent(_Trainer(mode="step", use_dp=True, dp_mechanism="opacus")),
        _no_data,
        stateless_trainers=(_Trainer,),
    )
    with pytest.raises(ValueError, match="Opacus"):
        pool.client("C1")


def test_a_global_model_must_cover_the_local_model():
    pool = SharedAgentPool(
        _fake_agent(_Trainer(mode="step")), _no_data, stateless_trainers=(_Trainer,)
    )
    pool.client("C1")  # builds the engine
    pool.check_covers({"weight": 1, "bias": 1, "running_mean": 1})
    with pytest.raises(RuntimeError, match="running_mean"):
        pool.check_covers({"weight": 1, "bias": 1})
