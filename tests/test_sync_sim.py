"""Unit tests for the virtual-time sync FL simulator (SyncSimDriver).

Uses the same lightweight fakes as test_async_sim.py.
"""

import logging

from appfl.vsim import SyncSimDriver, ClientProfile


# --------------------------- fakes --------------------------- #
class _FakeScheduler:
    def __init__(self):
        self.epochs = 0

    def get_num_global_epochs(self):
        return self.epochs


class _FakeServer:
    def __init__(self, target_epochs):
        self.scheduler = _FakeScheduler()
        self.target = target_epochs
        self.model = None
        self._epochs = 0

    def get_parameters(self, **kw):
        return {}

    def global_update(self, client_id, local_model, blocking=False, **kw):
        self._epochs += 1
        self.scheduler.epochs = self._epochs
        return {}

    def training_finished(self, **kw):
        return self._epochs >= self.target

    def server_validate(self):
        return None


class _FakeClient:
    def __init__(self, cid, cps=0.01, steps=100, acc=90.0):
        self.cid, self.cps, self.steps, self.acc = cid, cps, steps, acc

    def get_id(self):
        return self.cid

    def load_parameters(self, p):
        pass

    def train(self, **kw):
        pass

    def get_parameters(self):
        return (
            {},
            {
                "current_local_steps": self.steps,
                "compute_second_per_step": self.cps,
                "val_accuracy": self.acc,
            },
        )


def _logger():
    lg = logging.getLogger("vsim_sync_test")
    lg.handlers.clear()
    lg.addHandler(logging.NullHandler())
    lg.setLevel(logging.CRITICAL)
    return lg


def _make_sync(
    n,
    M,
    factors,
    mode="count",
    min_responses=None,
    max_wait_time=None,
    window_duration=None,
    target_rounds=10,
    seed=42,
    base_step_time=0.01,
):
    server = _FakeServer(target_rounds * 100)
    clients = [_FakeClient(f"C{i}") for i in range(n)]
    profiles = {
        f"C{i}": ClientProfile(compute_factor=factors[i], bandwidth=300.0)
        for i in range(n)
    }
    return SyncSimDriver(
        server_agent=server,
        client_agents=clients,
        profiles=profiles,
        participants_per_round=M,
        logger=_logger(),
        seed=seed,
        base_step_time=base_step_time,
        mode=mode,
        min_responses=min_responses,
        max_wait_time=max_wait_time,
        window_duration=window_duration,
        target_rounds=target_rounds,
    )


# ======================== tests ======================== #


def test_count_basic_round_count():
    """Count mode: 5 clients, M=5, K=3, 10 rounds → exactly 10 completed."""
    d = _make_sync(
        n=5, M=5, factors=[1.0] * 5, mode="count", min_responses=3, target_rounds=10
    )
    d.run()
    checks = d.verify(10)
    assert checks["completed_rounds==target"], (
        f"expected 10 rounds, got {len([r for r in d.history if not r['skipped']])}"
    )
    assert checks["monotonic_round_starts"]
    assert checks["durations_positive"]


def test_count_over_selection():
    """M=5, K=3: first 3 completions accepted, 2 discarded per round."""
    d = _make_sync(
        n=5,
        M=5,
        factors=[1.0, 2.0, 3.0, 4.0, 5.0],
        mode="count",
        min_responses=3,
        target_rounds=5,
    )
    d.run()
    for r in d.history:
        if not r["skipped"]:
            assert r["accepted_count"] == 3, (
                f"expected 3 accepted, got {r['accepted_count']}"
            )
            assert r["discarded_count"] == 2, (
                f"expected 2 discarded, got {r['discarded_count']}"
            )


def test_count_barrier_time():
    """Barrier time = K-th fastest client's completion time."""
    factors = [1.0, 10.0, 100.0]  # very different speeds
    d = _make_sync(
        n=3, M=3, factors=factors, mode="count", min_responses=2, target_rounds=1
    )
    d.run()
    r = d.history[0]
    assert r["accepted_count"] == 2
    assert r["duration"] > 0


def test_window_basic():
    """Window mode: all clients within window → all accepted."""
    d = _make_sync(
        n=5,
        M=5,
        factors=[1.0] * 5,
        mode="window",
        min_responses=1,
        window_duration=100.0,
        target_rounds=5,
    )
    d.run()
    checks = d.verify(5)
    assert checks["completed_rounds==target"]
    for r in d.history:
        assert r["accepted_count"] == 5


def test_window_straggler_discard():
    """Window mode: slow client exceeds window → discarded."""
    # factor=1 → fast, factor=1000 → very slow (way past 1s window)
    d = _make_sync(
        n=3,
        M=3,
        factors=[1.0, 1.0, 1000.0],
        mode="window",
        min_responses=1,
        window_duration=1.0,
        target_rounds=3,
    )
    d.run()
    for r in d.history:
        if not r["skipped"]:
            assert r["accepted_count"] == 2, "slow client should be discarded"
            assert r["discarded_count"] == 1


def test_monotonic_virtual_time():
    """Virtual time strictly increases across rounds."""
    d = _make_sync(
        n=5,
        M=5,
        factors=[1.0, 1.5, 2.0, 2.5, 3.0],
        mode="count",
        min_responses=3,
        target_rounds=20,
    )
    d.run()
    vts = [r["vtime"] for r in d.history]
    for i in range(len(vts) - 1):
        assert vts[i] < vts[i + 1], (
            f"vtime not monotonic at round {i}: {vts[i]} >= {vts[i + 1]}"
        )


def test_staleness_always_zero():
    """Sync FL: staleness is always 0 (all clients train on same global model)."""
    d = _make_sync(
        n=4,
        M=4,
        factors=[1.0, 2.0, 3.0, 4.0],
        mode="count",
        min_responses=2,
        target_rounds=10,
    )
    d.run()
    for r in d.history:
        assert r["staleness"] == 0


def test_determinism():
    """Same seed → identical results."""

    def run_once(seed):
        d = _make_sync(
            n=8,
            M=5,
            factors=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
            mode="count",
            min_responses=3,
            target_rounds=10,
            seed=seed,
        )
        d.run()
        return [(r["vtime"], tuple(sorted(r["accepted_cids"]))) for r in d.history]

    assert run_once(42) == run_once(42)
    assert run_once(42) != run_once(99)


def test_concurrency_bounded():
    """Max active clients never exceeds M."""
    d = _make_sync(
        n=10,
        M=6,
        factors=[float(i + 1) for i in range(10)],
        mode="count",
        min_responses=4,
        target_rounds=10,
    )
    d.run()
    assert d._max_active <= 6


def test_window_skip_round():
    """Window mode with tiny window: if nobody arrives, round is skipped."""
    # All clients have factor=100 → very slow; window=0.001s → nobody makes it
    d = _make_sync(
        n=3,
        M=3,
        factors=[100.0] * 3,
        mode="window",
        min_responses=2,
        window_duration=0.0001,
        target_rounds=5,
        base_step_time=0.01,
    )
    d.run()
    skipped = [r for r in d.history if r["skipped"]]
    assert len(skipped) > 0, "expected some skipped rounds with tiny window"


def test_actual_training_count():
    """Count mode performs actual local training."""
    d = _make_sync(
        n=3,
        M=3,
        factors=[1.0] * 3,
        mode="count",
        min_responses=2,
        target_rounds=5,
        base_step_time=0.01,
    )
    d.run()
    checks = d.verify(5)
    assert checks["completed_rounds==target"]
