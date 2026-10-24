"""Equivalence tests for flat-buffer aggregation (appfl.vsim.flat_aggregation).

Every scenario runs real APPFL agents twice: through the server agent
(``flat_aggregation=False``) and on the flat path, in-process or through a test
executor. The runs must produce the same history and bit-identical global models
with the same key order, at every global evaluation and at the end, and must leave
the server agent in the same state.
"""

import logging
import os

import pytest

torch = pytest.importorskip("torch")

from omegaconf import OmegaConf  # noqa: E402

from appfl.vsim import AsyncSimDriver, ClientProfile, SyncSimDriver  # noqa: E402
from appfl.vsim.codec import PackedState, pack_state, unpack_state  # noqa: E402
from appfl.vsim.executor import DeferredExecutor, TaskHandle  # noqa: E402
from appfl.vsim.flat_aggregation import FlatFedAsync, FlatFedAvg  # noqa: E402
from appfl.vsim.seeding import rng_guard  # noqa: E402

RESOURCES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")
IDS = [f"Client{i + 1}" for i in range(4)]
PROFILES = {
    cid: ClientProfile(compute_factor=1.0 + 0.5 * i) for i, cid in enumerate(IDS)
}
STEPS = 3

SCENARIOS = {
    "fedasync": dict(kind="async"),
    "fedasync-gradient": dict(kind="async", send_gradient=True),
    # VanillaTrainer returns parameters first, then buffers, unless optimize_memory
    # is off, when it returns the model's state_dict order.
    "fedasync-state_dict-order": dict(kind="async", trainer_memory=False),
    "fedasync-no-buffers": dict(kind="async", model="CNN"),
    "fedasync-sample_size": dict(kind="async", weights="sample_size"),
    "fedavg": dict(kind="sync"),
    "fedavg-state_dict-order": dict(kind="sync", trainer_memory=False),
    "fedavg-sample_size": dict(kind="sync", weights="sample_size"),
}


def _logger():
    lg = logging.getLogger("vsim_flat_test")
    lg.handlers.clear()
    lg.addHandler(logging.NullHandler())
    return lg


def _server(
    kind,
    tmp_dir,
    model="CNNBN",
    send_gradient=False,
    trainer_memory=True,
    weights="equal",
    server_cls=None,
):
    from appfl.agent import ServerAgent

    name = "server_fedasync.yaml" if kind == "async" else "server_fedavg.yaml"
    cfg = OmegaConf.load(os.path.join(RESOURCES, "configs", name))
    train = cfg.client_configs.train_configs
    train.num_local_steps = STEPS
    train.train_batch_size = 16
    train.do_validation = False
    train.do_pre_validation = False
    train.send_gradient = send_gradient
    train.optimize_memory = trainer_memory
    train.loss_fn_path = os.path.join(RESOURCES, "loss", "celoss.py")
    train.metric_path = os.path.join(RESOURCES, "metric", "acc.py")
    models = cfg.client_configs.model_configs
    models.model_path = os.path.join(
        RESOURCES, "model", "cnn_bn.py" if model == "CNNBN" else "cnn.py"
    )
    models.model_name = model
    models.model_kwargs.num_pixel = 16
    cfg.server_configs.num_clients = len(IDS)
    cfg.server_configs.num_global_epochs = 8 if kind == "async" else 3
    cfg.server_configs.aggregator_kwargs.client_weights_mode = weights
    cfg.server_configs.logging_output_dirname = tmp_dir
    server = (server_cls or ServerAgent)(server_agent_config=cfg)
    if weights == "sample_size":
        for i, cid in enumerate(IDS):
            server.set_sample_size(cid, 10 + 7 * i)
    return server


def _client(cid, context):
    from appfl.agent import ClientAgent

    cfg = OmegaConf.create(
        {
            "client_id": cid,
            "train_configs": {"device": "cpu", "logging_output_filename": ""},
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
    client = ClientAgent(client_agent_config=cfg)
    client.load_config(OmegaConf.create(context))
    return client


class _SerialExecutor(DeferredExecutor):
    """Trains each task as it is submitted; moves models as state dicts."""

    def __init__(self, clients):
        self.clients = {c.get_id(): c for c in clients}
        self.client_ids = list(self.clients)
        self._versions = {}
        self._results = {}
        self._next_tid = 0

    def setup(self, init_model, version, calibrate):
        self._versions[version] = init_model

    def expected_steps(self, cid):
        return STEPS

    def publish(self, version, params):
        self._versions[version] = params

    def submit(self, cid, version, params, deadline, seed):
        if params is not None:
            self._versions[version] = params
        handle = TaskHandle(self._next_tid, cid, version, deadline)
        self._next_tid += 1
        client = self.clients[cid]
        with rng_guard(seed):
            client.load_parameters(self._state(self._versions[version]))
            client.train()
            model, meta = client.get_parameters()
        model = {name: tensor.clone() for name, tensor in model.items()}
        self._results[handle.tid] = (self._wrap(model, version), dict(meta))
        return handle

    def _state(self, params):
        return params

    def _wrap(self, model, version):
        return model

    def result(self, handle):
        return self._results.pop(handle.tid)

    def discard(self, handle):
        self._results.pop(handle.tid, None)

    def shutdown(self):
        return {}


class _PackedExecutor(_SerialExecutor):
    """
    Moves models packed, as MPIExecutor does, and overwrites every buffer it is
    given back: a buffer still in use after recycle() corrupts the run.
    """

    accepts_packed = True

    def __init__(self, clients):
        super().__init__(clients)
        self.recycled = 0

    def setup(self, init_model, version, calibrate):
        assert isinstance(init_model, PackedState)
        super().setup(init_model, version, calibrate)

    def _state(self, params):
        assert isinstance(params, PackedState)
        return unpack_state(params.schema, params.buffer)

    def _wrap(self, model, version):
        # As a worker does: in the layout of the global model it trained from.
        return PackedState(*pack_state(model, layout=self._versions[version].schema))

    def result_packed(self, handle):
        return self.result(handle)

    def recycle(self, buffer):
        buffer.fill_(255)  # NaN as float32
        self.recycled += 1


class _ChangesOnRetrain:
    """A client whose local model changes structure from its second task on."""

    def __init__(self, client, change):
        self._client, self._change, self._calls = client, change, 0

    def __getattr__(self, name):
        return getattr(self._client, name)

    def get_parameters(self):
        model, meta = self._client.get_parameters()
        self._calls += 1
        return (self._change(model) if self._calls > 1 else model), meta


class _ParametersOnly:
    """A client that returns its named parameters only, no buffers."""

    def __init__(self, client):
        self._client = client

    def __getattr__(self, name):
        return getattr(self._client, name)

    def get_parameters(self):
        model, meta = self._client.get_parameters()
        names = {n for n, _ in self._client.trainer.model.named_parameters()}
        return {k: v for k, v in model.items() if k in names}, meta


def _run(scenario, tmp_dir, flat, executor_cls=None, wrap=None, eval_every=2):
    torch.set_num_threads(1)
    scenario = dict(scenario)
    kind = scenario.pop("kind")
    server = _server(kind, tmp_dir, **scenario)
    context = OmegaConf.to_container(server.get_client_configs(), resolve=True)
    clients = [_client(cid, context) for cid in IDS]
    if wrap is not None:
        clients = [wrap(c) for c in clients]
    executor = executor_cls(clients) if executor_cls is not None else None
    evals = []

    def validate():
        evals.append({k: v.clone() for k, v in server.model.state_dict().items()})
        return 0.0, 0.0

    server.server_validate = validate
    common = dict(
        logger=_logger(),
        seed=3,
        base_step_time=0.01,
        eval_every=eval_every,
        executor=executor,
        flat_aggregation=flat,
    )
    agents = None if executor is not None else clients
    if kind == "async":
        driver = AsyncSimDriver(server, agents, PROFILES, max_in_flight=3, **common)
    else:
        driver = SyncSimDriver(
            server,
            agents,
            PROFILES,
            participants_per_round=3,
            mode="count",
            min_responses=2,
            target_rounds=3,
            **common,
        )
    flat_at_start = driver._flat
    driver.run()
    return dict(
        driver=driver,
        server=server,
        executor=executor,
        evals=evals,
        flat_at_start=flat_at_start,
    )


def _assert_same(run, ref):
    assert run["driver"].history == ref["driver"].history
    server, ref_server = run["server"], ref["server"]
    model = server.get_parameters(serial_run=True)
    ref_model = ref_server.get_parameters(serial_run=True)
    assert list(model) == list(ref_model)
    for name in ref_model:
        assert torch.equal(model[name], ref_model[name]), name
    state, ref_state = server.model.state_dict(), ref_server.model.state_dict()
    for name in ref_state:
        assert torch.equal(state[name], ref_state[name]), name
    assert (
        server.scheduler.get_num_global_epochs()
        == ref_server.scheduler.get_num_global_epochs()
    )
    for attr in ("global_step", "client_step"):
        assert getattr(server.aggregator, attr, None) == getattr(
            ref_server.aggregator, attr, None
        )
    assert len(run["evals"]) == len(ref["evals"]) > 0
    for state, ref_state in zip(run["evals"], ref["evals"]):
        for name in ref_state:
            assert torch.equal(state[name], ref_state[name]), name


# ======================== tests ======================== #
@pytest.mark.parametrize("name", list(SCENARIOS))
def test_flat_path_matches_the_server_agent_in_process(name, tmp_path):
    ref = _run(SCENARIOS[name], str(tmp_path), flat=False)
    run = _run(SCENARIOS[name], str(tmp_path), flat=True)
    assert ref["flat_at_start"] is None
    expected = FlatFedAsync if SCENARIOS[name]["kind"] == "async" else FlatFedAvg
    assert isinstance(run["flat_at_start"], expected)
    assert run["driver"]._flat is run["flat_at_start"]  # stayed on the flat path
    _assert_same(run, ref)


@pytest.mark.parametrize("executor_cls", [_SerialExecutor, _PackedExecutor])
@pytest.mark.parametrize("name", ["fedasync", "fedasync-state_dict-order", "fedavg"])
def test_flat_path_through_an_executor_matches_the_server_agent(
    name, executor_cls, tmp_path
):
    ref = _run(SCENARIOS[name], str(tmp_path), flat=False)
    run = _run(SCENARIOS[name], str(tmp_path), flat=True, executor_cls=executor_cls)
    assert run["driver"]._flat is run["flat_at_start"] is not None
    _assert_same(run, ref)
    if executor_cls is _PackedExecutor:
        assert run["executor"].recycled > 0


@pytest.mark.parametrize("name", ["fedasync", "fedavg"])
def test_models_without_the_global_tensors_go_to_the_server_agent(name, tmp_path):
    """Clients that return parameters only: the first update leaves the flat path."""
    ref = _run(SCENARIOS[name], str(tmp_path), flat=False, wrap=_ParametersOnly)
    run = _run(SCENARIOS[name], str(tmp_path), flat=True, wrap=_ParametersOnly)
    assert run["flat_at_start"] is not None and run["driver"]._flat is None
    _assert_same(run, ref)


def _with_extra(model):
    return {**model, "extra": torch.ones(3)}


def test_async_leaves_the_flat_path_mid_run_without_changing_results(tmp_path):
    def wrap(client):
        return _ChangesOnRetrain(client, _with_extra)

    ref = _run(SCENARIOS["fedasync"], str(tmp_path), flat=False, wrap=wrap)
    run = _run(SCENARIOS["fedasync"], str(tmp_path), flat=True, wrap=wrap)
    assert run["flat_at_start"].updates > 0 and run["driver"]._flat is None
    _assert_same(run, ref)


def test_sync_refuses_a_structure_change_after_flat_rounds(tmp_path):
    with pytest.raises(RuntimeError, match="flat_aggregation=False"):
        _run(
            SCENARIOS["fedavg"],
            str(tmp_path),
            flat=True,
            wrap=lambda c: _ChangesOnRetrain(c, _with_extra),
        )


def test_flat_path_stays_off_for_a_customized_server_agent(tmp_path):
    from appfl.agent import ServerAgent

    class Customized(ServerAgent):
        def global_update(self, *args, **kwargs):
            return super().global_update(*args, **kwargs)

    run = _run(
        dict(SCENARIOS["fedasync"], server_cls=Customized), str(tmp_path), flat=True
    )
    assert run["flat_at_start"] is None
