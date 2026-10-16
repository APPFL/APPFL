"""Equivalence tests for vsim drivers driven through a DeferredExecutor.

A deferred executor trains clients outside the driver, in whatever physical order it
likes, while the driver computes every duration at dispatch. These tests run each
scenario twice, in-process and through a test double that trains pending tasks in a
seeded random order, and require identical histories, identical global updates and
identical per-task random draws.
"""

import logging
import random

import pytest

from appfl.vsim import AsyncSimDriver, ClientProfile, SyncSimDriver
from appfl.vsim.executor import DeferredExecutor, TaskHandle
from appfl.vsim.seeding import rng_guard

STEP = 0.01  # seconds per step; 100 steps -> 1.0 virtual second per compute factor


# --------------------------- fakes --------------------------- #
class _Scheduler:
    def __init__(self):
        self.epochs = 0
        self.num_clients = 99

    def get_num_global_epochs(self):
        return self.epochs

    def set_num_clients(self, num_clients):
        self.num_clients = num_clients


class _Server:
    """Records every update; tracks download versions like FedAsyncAggregator."""

    def __init__(self, target):
        self.scheduler = _Scheduler()
        self.target = target
        self.model = None
        self.client_step = {}
        self.updates = []

    def get_parameters(self, **kw):
        if kw.get("client_id") is not None:
            self.client_step[kw["client_id"]] = self.scheduler.epochs
        return {"w": float(self.scheduler.epochs)}

    def global_update(self, client_id, local_model, blocking=False, **kw):
        staleness = self.scheduler.epochs - self.client_step.get(client_id, 0)
        self.updates.append(
            (client_id, local_model["w"], kw["draw"], kw.get("virtual_time"), staleness)
        )
        self.scheduler.epochs += 1
        return {"w": float(self.scheduler.epochs)}

    def training_finished(self, **kw):
        return self.scheduler.epochs >= self.target

    def server_validate(self):
        return None


class _Client:
    """Trains by drawing from the global RNG, so task seeding is observable."""

    def __init__(self, cid, steps=100):
        self.cid, self.steps = cid, steps
        self.loaded = None
        self.trained = 0

    def get_id(self):
        return self.cid

    def load_parameters(self, params):
        self.loaded = dict(params)

    def train(self, **kw):
        self.trained += 1
        self.draw = random.random()

    def get_parameters(self):
        meta = {
            "current_local_steps": self.steps,
            "compute_second_per_step": 0.5,  # must be ignored: base_step_time wins
            "draw": self.draw,
        }
        return {"w": self.loaded["w"] + self.draw}, meta


class _ShuffledExecutor(DeferredExecutor):
    """
    Trains fake clients in-process, in a seeded random physical order.

    Like MPI workers, it trains tasks ahead of need: whenever a result is requested,
    a random subset of the other pending tasks is trained first, in random order.
    """

    def __init__(self, clients, shuffle_seed=0, step_time=None):
        self.clients = {c.get_id(): c for c in clients}
        self.client_ids = list(self.clients)
        self._calibrated = step_time
        self.step_time = None
        self._rng = random.Random(shuffle_seed)
        self._versions = {}
        self._pending = {}
        self._results = {}
        self._next_tid = 0
        self.setup_calls = []
        self.submitted = []
        self.seeds = []
        self.result_order = []
        self.discarded = []

    def setup(self, init_model, version, calibrate):
        self.setup_calls.append((version, calibrate))
        self._versions[version] = init_model
        if calibrate:
            self.step_time = self._calibrated

    def expected_steps(self, cid):
        return 100

    def publish(self, version, params):
        self._versions[version] = params

    def submit(self, cid, version, params, deadline, seed):
        if params is not None:
            self._versions[version] = params
        assert version in self._versions, "submitted an unpublished version"
        handle = TaskHandle(self._next_tid, cid, version, deadline)
        self._next_tid += 1
        self._pending[handle.tid] = (handle, seed)
        self.submitted.append(handle)
        self.seeds.append(seed)
        return handle

    def _train(self, tid):
        handle, seed = self._pending.pop(tid)
        client = self.clients[handle.cid]
        with rng_guard(seed):
            client.load_parameters(self._versions[handle.version])
            client.train()
            self._results[tid] = client.get_parameters()

    def result(self, handle):
        self.result_order.append(handle)
        if handle.tid not in self._results:
            others = [tid for tid in self._pending if tid != handle.tid]
            self._rng.shuffle(others)
            for tid in others[: self._rng.randint(0, len(others))]:
                self._train(tid)
            self._train(handle.tid)
        return self._results.pop(handle.tid)

    def discard(self, handle):
        self.discarded.append(handle)
        self._pending.pop(handle.tid, None)
        self._results.pop(handle.tid, None)

    def shutdown(self):
        return {}


class _OfflineUntil(ClientProfile):
    def __init__(self, until, **kw):
        super().__init__(**kw)
        self.until = until

    def next_available(self, vtime):
        return max(vtime, self.until)


def _logger():
    lg = logging.getLogger("vsim_deferred_test")
    lg.handlers.clear()
    lg.addHandler(logging.NullHandler())
    lg.setLevel(logging.CRITICAL)
    return lg


def _profiles(factors, offline=None):
    profiles = {}
    for i, f in enumerate(factors):
        cid = f"C{i}"
        if offline and cid in offline:
            profiles[cid] = _OfflineUntil(offline[cid], compute_factor=f)
        else:
            profiles[cid] = ClientProfile(compute_factor=f)
    return profiles


def _async(factors, deferred, K=3, target=30, seed=5, offline=None, **kw):
    clients = [_Client(f"C{i}") for i in range(len(factors))]
    executor = _ShuffledExecutor(clients, shuffle_seed=seed) if deferred else None
    server = _Server(target)
    driver = AsyncSimDriver(
        server,
        None if deferred else clients,
        _profiles(factors, offline),
        max_in_flight=K,
        logger=_logger(),
        seed=seed,
        base_step_time=kw.pop("base_step_time", STEP),
        executor=executor,
        **kw,
    )
    return driver, server, executor


def _sync(factors, deferred, seed=5, target_rounds=6, **kw):
    clients = [_Client(f"C{i}") for i in range(len(factors))]
    executor = _ShuffledExecutor(clients, shuffle_seed=seed) if deferred else None
    server = _Server(10_000)
    driver = SyncSimDriver(
        server_agent=server,
        client_agents=None if deferred else clients,
        profiles=_profiles(factors),
        logger=_logger(),
        seed=seed,
        base_step_time=STEP,
        target_rounds=target_rounds,
        executor=executor,
        **kw,
    )
    return driver, server, executor


def _run_both(build, *args, **kw):
    runs = []
    for deferred in (False, True):
        driver, server, executor = build(*args, deferred=deferred, **kw)
        driver.run()
        runs.append((driver, server, executor))
    return runs


# ======================== async ======================== #
@pytest.mark.parametrize("shuffle_seed", [0, 1, 2])
def test_async_deferred_matches_in_process(shuffle_seed):
    factors = [1.0, 3.0, 0.5, 2.0, 1.5, 4.0]
    (eager, s_eager, _), (deferred, s_deferred, ex) = _run_both(
        _async, factors, K=3, target=30, seed=shuffle_seed, offline={"C2": 6.0}
    )
    assert deferred.history == eager.history
    assert s_deferred.updates == s_eager.updates  # same models, draws and staleness
    assert [r["staleness"] for r in deferred.history] == [
        u[4] for u in s_deferred.updates
    ]
    assert any(r["cid"] == "C2" for r in deferred.history)  # offline client ran
    assert ex.setup_calls == [(0, False)]


def test_async_deferred_submits_dispatch_version_and_deadline():
    driver, server, ex = _async([1.0, 2.0, 0.5, 1.5], deferred=True, K=2, target=12)
    driver.run()
    assert len(ex.result_order) == len(driver.history)
    for record, handle in zip(driver.history, ex.result_order):
        assert handle.cid == record["cid"]
        assert handle.deadline == pytest.approx(record["vtime"])
        assert handle.version == record["epoch"] - 1 - record["staleness"]


def test_async_results_are_consumed_in_deadline_order():
    """The MPI coordinator prefetches in (deadline, tid) order; the driver must
    consume in that same order."""
    driver, _, ex = _async([1.0, 3.0, 0.5, 2.0, 1.5, 4.0], deferred=True, target=25)
    driver.run()
    keys = [(h.deadline, h.tid) for h in ex.result_order]
    assert keys == sorted(keys)


def test_async_leftover_tasks_are_discarded():
    driver, _, ex = _async([1.0, 2.0, 3.0, 4.0], deferred=True, K=3, target=10)
    driver.run()
    consumed = {h.tid for h in ex.result_order}
    # The last completion's replacement is queued but never dispatched, so
    # max_in_flight - 1 submitted tasks are still outstanding when the run ends.
    assert len(ex.discarded) == 2
    assert {h.tid for h in ex.discarded} == {
        p["handle"].tid for p in driver._pending.values()
    }
    assert not consumed & {h.tid for h in ex.discarded}


def test_async_steps_mismatch_raises():
    driver, _, ex = _async([1.0, 2.0], deferred=True, K=2, target=4)
    ex.clients["C1"].steps = 50
    with pytest.raises(RuntimeError, match="computed at dispatch from 100 steps"):
        driver.run()


def test_deferred_without_step_time_raises():
    with pytest.raises(RuntimeError, match="must be known when the client is"):
        _async([1.0, 2.0], deferred=True, base_step_time=None)


def test_calibrated_step_time_drives_durations():
    clients = [_Client(f"C{i}") for i in range(3)]
    ex = _ShuffledExecutor(clients, step_time=0.02)
    driver = AsyncSimDriver(
        _Server(9),
        None,
        _profiles([1.0, 2.0, 3.0]),
        max_in_flight=2,
        logger=_logger(),
        base_step_time=None,
        executor=ex,
    )
    assert ex.setup_calls == [(0, True)]
    assert driver.base_step_time == 0.02
    driver.run()
    eager, _, _ = _async(
        [1.0, 2.0, 3.0], deferred=False, K=2, target=9, seed=42, base_step_time=0.02
    )
    eager.run()
    assert driver.history == eager.history


def test_task_seeding_can_be_disabled():
    seeded, _, ex_seeded = _async([1.0, 2.0], deferred=True, K=2, target=6)
    seeded.run()
    assert all(isinstance(s, int) for s in ex_seeded.seeds)
    assert len(set(ex_seeded.seeds)) == len(ex_seeded.seeds)

    unseeded, _, ex_unseeded = _async(
        [1.0, 2.0], deferred=True, K=2, target=6, task_seeding=False
    )
    unseeded.run()
    assert ex_unseeded.seeds and all(s is None for s in ex_unseeded.seeds)


def test_a_clients_draws_do_not_depend_on_concurrency():
    """With task seeding, a client's k-th task draws the same numbers whatever
    else ran before it, here varied through max_in_flight."""

    def draws_per_client(K):
        driver, server, _ = _async([1.0, 2.0, 0.5, 1.5], deferred=False, K=K, target=20)
        driver.run()
        per_client = {}
        for cid, _, draw, _, _ in server.updates:
            per_client.setdefault(cid, []).append(draw)
        return per_client

    narrow, wide = draws_per_client(1), draws_per_client(4)
    for cid in narrow:
        n = min(len(narrow[cid]), len(wide[cid]))
        assert n > 0 and narrow[cid][:n] == wide[cid][:n]


# ======================== sync ======================== #
@pytest.mark.parametrize("shuffle_seed", [0, 3])
def test_sync_count_over_selection_with_skips_matches_in_process(shuffle_seed):
    # Only C0-C2 finish within max_wait_time, so a round reaches its quorum of 3
    # only when all three fast clients are selected; other rounds are skipped.
    factors = [1.0, 1.0, 1.5, 5.0, 5.0, 9.0]
    (eager, s_eager, _), (deferred, s_deferred, ex) = _run_both(
        _sync,
        factors,
        seed=shuffle_seed,
        participants_per_round=5,
        mode="count",
        min_responses=3,
        max_wait_time=2.0,
        target_rounds=6,
    )
    assert deferred.history == eager.history
    assert s_deferred.updates == s_eager.updates
    assert any(r["skipped"] for r in deferred.history)
    assert any(not r["skipped"] for r in deferred.history)
    # Only accepted clients are trained remotely; the in-process engine trains all.
    accepted = sum(r["accepted_count"] for r in deferred.history)
    assert len(ex.submitted) == accepted
    assert sum(c.trained for c in ex.clients.values()) == accepted


def test_sync_window_extension_matches_in_process():
    # Window closes at 1.2 with two arrivals; quorum 3 is met at 2.0 (< max_wait).
    factors = [1.0, 1.1, 2.0, 2.5, 6.0]
    (eager, s_eager, _), (deferred, s_deferred, _) = _run_both(
        _sync,
        factors,
        participants_per_round=5,
        mode="window",
        min_responses=3,
        window_duration=1.2,
        max_wait_time=4.0,
        target_rounds=4,
    )
    assert deferred.history == eager.history
    assert s_deferred.updates == s_eager.updates
    assert all(
        r["t_barrier"] - r["t_start"] == pytest.approx(2.0) for r in eager.history
    )


def test_sync_publishes_once_per_round_and_trains_in_arrival_order():
    driver, _, ex = _sync(
        [1.0, 2.0, 3.0, 4.0],
        deferred=True,
        participants_per_round=4,
        mode="count",
        min_responses=2,
        target_rounds=3,
    )
    published = []
    original = ex.publish

    def publish(version, params):
        published.append(version)
        original(version, params)

    ex.publish = publish
    driver.run()
    assert len(published) == 3  # one broadcast per round, not one per client
    assert all(h.version in published for h in ex.submitted)
    # Results are consumed in arrival order, matching the aggregation order.
    for r, start in zip(driver.history, range(0, len(ex.result_order), 2)):
        consumed = ex.result_order[start : start + 2]
        assert [h.cid for h in consumed] == r["accepted_cids"]
        assert [h.deadline for h in consumed] == sorted(h.deadline for h in consumed)
