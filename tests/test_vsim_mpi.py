"""MPI tests for the multi-node virtual-time simulator (appfl.vsim.mpi).

Run with, e.g., ``mpirun -n 3 python -m pytest --with-mpi tests/test_vsim_mpi.py``.
Rank 0 runs each scenario in-process as a reference and again through an
MPIExecutor; every other rank serves it as a worker. The two runs must produce the
same history and bit-identical global models (CPU, one thread per rank), whichever
worker trains each task.
"""

import functools
import gc
import logging
import os
import sys
import time
import traceback

import pytest

pytest.importorskip("mpi4py")
torch = pytest.importorskip("torch")

from mpi4py import MPI  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from appfl.vsim import (  # noqa: E402
    AsyncSimDriver,
    ClientProfile,
    SharedAgentPool,
    SyncSimDriver,
    load_client_datasets,
)
from appfl.vsim.mpi import MPIExecutor, run_worker  # noqa: E402
from appfl.vsim.mpi.coordinator import assign_homes  # noqa: E402

pytestmark = pytest.mark.mpi(min_size=2)

RESOURCES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")


# --------------------------- fakes --------------------------- #
class _Scheduler:
    def __init__(self):
        self.epochs = 0
        self.num_clients = 99

    def get_num_global_epochs(self):
        return self.epochs

    def set_num_clients(self, num_clients):
        self.num_clients = num_clients


class _TensorServer:
    """Averages tensor models; sync mode aggregates once the round's count is in."""

    def __init__(self, target, sync=False):
        self.scheduler = _Scheduler()
        self.target, self.sync = target, sync
        self.model = None
        self.w = torch.zeros(4096)
        self.buffer = []
        self.updates = []
        self.ranks = []  # (client, rank that trained the update), MPI runs only

    def get_parameters(self, **kw):
        return {"w": self.w.clone(), "epoch": torch.tensor(self.scheduler.epochs)}

    def global_update(self, client_id, local_model, blocking=False, **kw):
        self.updates.append((client_id, float(local_model["w"].sum()), kw["draw"]))
        self.ranks.append((client_id, kw.get("rank")))
        if self.sync:
            self.buffer.append(local_model["w"].clone())
            if len(self.buffer) == self.scheduler.num_clients:
                self.w = torch.stack(self.buffer).mean(0)
                self.buffer = []
                self.scheduler.epochs += 1
        else:
            self.w = 0.5 * self.w + 0.5 * local_model["w"]
            self.scheduler.epochs += 1
        return {"w": self.w.clone()}

    def training_finished(self, **kw):
        return self.scheduler.epochs >= self.target

    def server_validate(self):
        return None


class _TensorClient:
    """Trains by adding seeded noise, so per-task seeding is observable."""

    def __init__(self, cid, step_time=0.004, rank=None, delay=0.0):
        self.cid, self.step_time = cid, step_time
        self.rank, self.delay = rank, delay

    def get_id(self):
        return self.cid

    def expected_local_steps(self):
        return 100

    def load_parameters(self, params):
        self.w = params["w"].clone()

    def train(self, **kw):
        if self.delay:
            time.sleep(self.delay)
        self.draw = float(torch.rand(1))
        self.w = self.w + torch.randn(self.w.shape)

    def get_parameters(self):
        meta = {
            "current_local_steps": 100,
            "compute_second_per_step": self.step_time,
            "draw": self.draw,
        }
        if self.rank is not None:
            meta["rank"] = self.rank  # the drivers pass it on to global_update
        return {"w": self.w}, meta


def _make_tensor_client(cid, context, device):
    return _TensorClient(cid, rank=MPI.COMM_WORLD.Get_rank())


def _make_slow_rank1_client(cid, context, device):
    rank = MPI.COMM_WORLD.Get_rank()
    return _TensorClient(cid, rank=rank, delay=0.03 if rank == 1 else 0.0)


def _logger():
    lg = logging.getLogger("vsim_mpi_test")
    lg.handlers.clear()
    lg.addHandler(logging.NullHandler())
    return lg


IDS = [f"Client{i + 1}" for i in range(9)]
PROFILES = {
    cid: ClientProfile(compute_factor=0.5 + (i % 4) * 0.5) for i, cid in enumerate(IDS)
}
# Expected durations up to a common factor (every client runs 100 steps).
DURATIONS = {cid: profile.compute_factor for cid, profile in PROFILES.items()}


def _tensor_driver(mode, executor, base_step_time=0.01, target_epochs=None):
    clients = None if executor else [_TensorClient(cid) for cid in IDS]
    if mode == "async":
        server = _TensorServer(30)
        driver = AsyncSimDriver(
            server,
            clients,
            PROFILES,
            max_in_flight=5,
            logger=_logger(),
            seed=7,
            base_step_time=base_step_time,
            executor=executor,
            target_epochs=target_epochs,
        )
    else:
        server = _TensorServer(10**6, sync=True)
        driver = SyncSimDriver(
            server,
            clients,
            PROFILES,
            participants_per_round=6,
            logger=_logger(),
            seed=7,
            base_step_time=base_step_time,
            mode="count",
            min_responses=4,
            target_rounds=6,
            executor=executor,
        )
    return driver, server


def _on_rank0(comm, run, make_client):
    """
    Rank 0 runs ``run(executor_factory)``; the other ranks serve as workers.

    ``run`` gets a factory for MPIExecutors and returns its result after shutting
    each executor down. Any exception before that aborts the job, because the
    workers would otherwise wait forever.
    """
    if comm.Get_rank() != 0:
        for _ in range(run.executors):
            run_worker(comm, make_client)
        return None
    try:
        return run(lambda ids, **kw: MPIExecutor(comm, ids, stall_timeout=120, **kw))
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        comm.Abort(1)


def _executors(count):
    def mark(fn):
        fn.executors = count
        return fn

    return mark


def _trained(stats):
    """Tasks trained per worker rank."""
    return {int(rank): s["tasks"] for rank, s in stats["worker_stats"].items()}


PLACEMENTS = [
    pytest.param(dict(placement="dynamic"), id="dynamic"),
    pytest.param(dict(placement="dynamic", queue_depth=2), id="dynamic-depth2"),
    pytest.param(dict(placement="static"), id="static"),
]


# ======================== tests ======================== #
@pytest.mark.parametrize("placement", PLACEMENTS)
@pytest.mark.parametrize("balanced", [False, True])
@pytest.mark.parametrize("mode", ["async", "sync"])
def test_mpi_run_matches_in_process(mode, balanced, placement):
    """``balanced`` chooses homes by expected load and, in async mode, declares
    the target so the executor drops tasks that can never be consumed."""
    torch.set_num_threads(1)
    homes_kw = {}
    if balanced:
        homes_kw = dict(durations=DURATIONS, mode=mode, max_in_flight=5)

    @_executors(1)
    def run(new_executor):
        reference, ref_server = _tensor_driver(mode, None)
        reference.run()
        # A tiny prefetch budget exercises the on-demand fetch path.
        executor = new_executor(IDS, prefetch_mb=0.05, **homes_kw, **placement)
        driver, server = _tensor_driver(
            mode, executor, target_epochs=30 if balanced else None
        )
        driver.run()
        stats = executor.shutdown()
        return reference, ref_server, driver, server, stats, executor.homes

    out = _on_rank0(MPI.COMM_WORLD, run, _make_tensor_client)
    if out is None:
        return
    reference, ref_server, driver, server, stats, homes = out
    assert driver.history == reference.history
    assert server.updates == ref_server.updates
    assert torch.equal(server.w, ref_server.w)
    assert stats["results"] == len(server.updates)
    workers = MPI.COMM_WORLD.Get_size() - 1
    assert homes == assign_homes(IDS, workers, **homes_kw)
    assert stats["trained"] <= stats["tasks"] - stats["dropped_unsent"]
    held = [s["max_held"] for s in stats["worker_stats"].values()]
    if placement["placement"] == "static":
        assert all(rank == homes[cid] for cid, rank in server.ranks)
        assert stats["assigned_away"] == 0
    else:
        assert max(held) <= placement.get("queue_depth", 1)
    if mode == "async":
        assert (stats["dropped"] > 0) == balanced
    if mode == "sync":
        # One broadcast per new version; no per-client model sends.
        assert stats["model_sends"] == 0 and stats["broadcasts"] >= 1
        assert stats["dropped"] == 0


def test_mpi_dynamic_placement_works_around_a_slow_worker():
    """Rank 1 needs 30 ms more per task. Static placement trains every task on its
    client's home; dynamic placement gives rank 1 the fewest tasks."""
    if MPI.COMM_WORLD.Get_size() < 3:
        pytest.skip("needs at least two workers")
    torch.set_num_threads(1)

    @_executors(2)
    def run(new_executor):
        reference, ref_server = _tensor_driver("async", None)
        reference.run()
        out = {}
        for placement in ("static", "dynamic"):
            executor = new_executor(IDS, placement=placement)
            driver, server = _tensor_driver("async", executor)
            driver.run()
            stats = executor.shutdown()
            out[placement] = (driver.history, server, stats, executor.homes)
        return reference.history, ref_server.updates, out

    result = _on_rank0(MPI.COMM_WORLD, run, _make_slow_rank1_client)
    if result is None:
        return
    ref_history, ref_updates, out = result
    for placement, (history, server, _, _) in out.items():
        assert history == ref_history, placement
        assert server.updates == ref_updates, placement
    _, server, stats, homes = out["static"]
    assert all(rank == homes[cid] for cid, rank in server.ranks)
    _, server, stats, _ = out["dynamic"]
    trained = _trained(stats)
    assert trained[1] < min(n for rank, n in trained.items() if rank != 1)
    assert stats["assigned_away"] > 0


def test_mpi_pinned_clients_train_on_their_home():
    torch.set_num_threads(1)
    pinned = IDS[::3]

    @_executors(1)
    def run(new_executor):
        reference, ref_server = _tensor_driver("sync", None)
        reference.run()
        executor = new_executor(IDS, pinned=pinned)
        driver, server = _tensor_driver("sync", executor)
        driver.run()
        executor.shutdown()
        return reference.history, ref_server.updates, driver.history, server, executor

    out = _on_rank0(MPI.COMM_WORLD, run, _make_tensor_client)
    if out is None:
        return
    ref_history, ref_updates, history, server, executor = out
    assert history == ref_history
    assert server.updates == ref_updates
    assert executor.pinned == frozenset(pinned)
    trained_pinned = [(cid, rank) for cid, rank in server.ranks if cid in pinned]
    assert trained_pinned
    assert all(rank == executor.homes[cid] for cid, rank in trained_pinned)


def test_mpi_calibrates_step_time():
    @_executors(1)
    def run(new_executor):
        executor = new_executor(IDS)
        driver, _ = _tensor_driver("async", executor, base_step_time=None)
        step_time = driver.base_step_time
        driver.run()
        executor.shutdown()
        return step_time, executor.step_time

    out = _on_rank0(MPI.COMM_WORLD, run, _make_tensor_client)
    if out is None:
        return
    step_time, calibrated = out
    assert step_time == calibrated == pytest.approx(0.004)


# ------------------ real APPFL agents ------------------ #
def _server_config(name, tmp_dir, num_clients, epochs):
    from appfl.agent import ServerAgent

    cfg = OmegaConf.load(os.path.join(RESOURCES, "configs", name))
    train = cfg.client_configs.train_configs
    train.num_local_steps = 3
    train.train_batch_size = 16
    train.do_validation = False
    train.do_pre_validation = False
    train.loss_fn_path = os.path.join(RESOURCES, "loss", "celoss.py")
    train.metric_path = os.path.join(RESOURCES, "metric", "acc.py")
    cfg.client_configs.model_configs.model_path = os.path.join(
        RESOURCES, "model", "cnn.py"
    )
    cfg.client_configs.model_configs.model_kwargs.num_pixel = 16
    cfg.server_configs.num_clients = num_clients
    cfg.server_configs.num_global_epochs = epochs
    cfg.server_configs.logging_output_dirname = tmp_dir
    return ServerAgent(server_agent_config=cfg)


def _real_client_config(cid, device):
    index = int(cid.replace("Client", "")) - 1
    return OmegaConf.create(
        {
            "client_id": cid,
            "train_configs": {"device": device, "logging_output_filename": ""},
            "data_configs": {
                "dataset_path": os.path.join(
                    RESOURCES, "dataset", "synthetic_dataset.py"
                ),
                "dataset_name": "get_synthetic",
                "dataset_kwargs": {"num_clients": 4, "client_id": index},
            },
        }
    )


def _make_real_client(cid, context, device):
    from appfl.agent import ClientAgent

    client = ClientAgent(client_agent_config=_real_client_config(cid, device))
    client.load_config(OmegaConf.create(context))
    return client


def _load_real_data(cid, context, device):
    return load_client_datasets(_real_client_config(cid, device))


def _record_rounds(server):
    """Record the trainer round counter that reaches each global update."""
    rounds = []
    update = server.global_update

    def recording(client_id, local_model, blocking=False, **kw):
        rounds.append((client_id, kw.get("round")))
        return update(
            client_id=client_id, local_model=local_model, blocking=blocking, **kw
        )

    server.global_update = recording
    return rounds


def _real_driver(kind, tmp_dir, executor):
    ids = [f"Client{i + 1}" for i in range(4)]
    profiles = {cid: ClientProfile(compute_factor=1.0 + i) for i, cid in enumerate(ids)}
    if kind == "fedasync":
        server = _server_config("server_fedasync.yaml", tmp_dir, 4, 8)
    else:
        server = _server_config("server_fedavg.yaml", tmp_dir, 4, 3)
    context = OmegaConf.to_container(server.get_client_configs(), resolve=True)
    clients = None
    if executor is None:
        clients = [_make_real_client(cid, context, "cpu") for cid in ids]
    else:
        executor = executor(
            ids,
            context=context,
            durations={cid: p.compute_factor for cid, p in profiles.items()},
            mode="async" if kind == "fedasync" else "sync",
            max_in_flight=3,
        )
    common = dict(logger=_logger(), seed=3, base_step_time=0.01, executor=executor)
    if kind == "fedasync":
        # AsyncScheduler counts one epoch per arrival, so the executor may drop
        # tasks past the 8th arrival.
        driver = AsyncSimDriver(
            server, clients, profiles, max_in_flight=3, target_epochs=8, **common
        )
    else:
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
    rounds = _record_rounds(server)
    # APPFL's frequent gc.collect() calls are slow over a large heap; MPIExecutor
    # freezes after setup, so do the same for the in-process reference.
    gc.collect()
    gc.freeze()
    return driver, server, executor, rounds


@pytest.mark.parametrize("clients", ["full", "shared"])
def test_mpi_real_agents_match_in_process(tmp_path, clients):
    """Workers hold full ClientAgents under static placement, or serve every client
    from one shared ClientAgent under dynamic placement; the reference trains full
    ClientAgents in-process."""
    torch.set_num_threads(1)
    placement = "static" if clients == "full" else "dynamic"

    @_executors(2)
    def run(new_executor):
        results = {}
        for kind in ("fedasync", "fedavg"):
            reference, ref_server, _, ref_rounds = _real_driver(
                kind, str(tmp_path), None
            )
            reference.run()
            driver, server, executor, rounds = _real_driver(
                kind,
                str(tmp_path),
                functools.partial(new_executor, placement=placement),
            )
            driver.run()
            executor.shutdown()
            results[kind] = (
                reference.history,
                driver.history,
                ref_server.get_parameters(serial_run=True),
                server.get_parameters(serial_run=True),
                ref_rounds,
                rounds,
            )
        return results

    if clients == "shared":
        make_client = SharedAgentPool(_make_real_client, _load_real_data)
    else:
        make_client = _make_real_client
    out = _on_rank0(MPI.COMM_WORLD, run, make_client)
    if out is None:
        return
    for kind, result in out.items():
        ref_history, history, ref_model, model, ref_rounds, rounds = result
        assert history == ref_history, kind
        assert list(model) == list(ref_model), kind
        for name in model:
            assert torch.equal(model[name], ref_model[name]), (kind, name)
        if kind == "fedasync":
            # Every selected client trains in both runs, so a client's round
            # counter matches even when its tasks moved between workers.
            assert rounds == ref_rounds
