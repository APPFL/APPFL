"""Unit tests for the virtual-time async FL simulator (appfl.simulator).

These test the core event-queue logic in isolation using lightweight fakes for
ServerAgent / ClientAgent, so no dataset, model, or real training is needed.
"""

import logging
from collections import Counter

from appfl.simulator import AsyncSimDriver, ClientProfile


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

    def get_parameters(self, **kw):
        return {}  # no tensors -> model_bytes == 0 -> comm time 0

    def global_update(self, client_id, local_model, blocking=False, **kw):
        self.scheduler.epochs += 1
        return {}  # dict (not a Future), like AsyncScheduler+FedAsync

    def training_finished(self, **kw):
        return self.scheduler.epochs >= self.target

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


class _CalibrationClient(_FakeClient):
    def __init__(self, cid, cps):
        super().__init__(cid, cps=cps, steps=3)
        self.train_calls = 0

    def train(self, **kw):
        self.train_calls += 1


def _silent_logger():
    lg = logging.getLogger("vsim_test")
    lg.handlers.clear()
    lg.addHandler(logging.NullHandler())
    lg.setLevel(logging.CRITICAL)
    return lg


def _make_driver(
    n, K, factors, target=20, seed=42, base_step_time=0.01, driver_cls=AsyncSimDriver
):
    server = _FakeServer(target)
    clients = [_FakeClient(f"C{i}") for i in range(n)]
    profiles = {
        f"C{i}": ClientProfile(compute_factor=factors[i], bandwidth=300.0)
        for i in range(n)
    }
    return driver_cls(
        server,
        clients,
        profiles,
        max_concurrency=K,
        logger=_silent_logger(),
        seed=seed,
        base_step_time=base_step_time,
    )


# --------------------------- tests --------------------------- #
def test_client_profile_math():
    p = ClientProfile(compute_factor=2.0, bandwidth=100.0)
    assert p.compute_time(0.01, 100) == 2.0
    assert p.comm_time(0) == 0.0
    one_mib = 1024 * 1024
    # bytes*8/1MiB/bw*2 = 8/100*2 = 0.16
    assert abs(p.comm_time(one_mib) - 0.16) < 1e-9
    assert abs(p.duration(0.01, 100, one_mib) - 2.16) < 1e-9
    assert p.available(123.0) is True


def test_virtual_time_monotonic_and_count():
    d = _make_driver(4, 2, [1, 1, 1, 1], target=20)
    d.run()
    vts = [r["vtime"] for r in d.history]
    assert vts == sorted(vts)  # virtual time never goes backward
    assert len(d.history) == 20  # exactly target completions


def test_staleness_nonnegative():
    d = _make_driver(4, 2, [1, 2, 3, 4], target=20)
    d.run()
    assert all(r["staleness"] >= 0 for r in d.history)
    assert all(isinstance(r["staleness"], int) for r in d.history)


def test_determinism_with_fixed_step_time():
    def seq(d):
        d.run()
        return [(round(r["vtime"], 6), r["cid"], r["staleness"]) for r in d.history]

    a = seq(_make_driver(4, 2, [1.0, 2.0, 0.5, 1.5], target=24, seed=7))
    b = seq(_make_driver(4, 2, [1.0, 2.0, 0.5, 1.5], target=24, seed=7))
    assert a == b  # same seed + fixed step time -> bit-exact


def test_heterogeneity_fast_completes_more():
    # C0 fast (factor 0.5), C1 slow (factor 4.0); fast should complete more often
    d = _make_driver(2, 2, [0.5, 4.0], target=30, seed=1)
    d.run()
    c = Counter(r["cid"] for r in d.history)
    assert c["C0"] > c["C1"]


def test_concurrency_never_exceeds_K():
    class _CapDriver(AsyncSimDriver):
        def _dispatch_idle(self):
            super()._dispatch_idle()
            self._maxK = max(getattr(self, "_maxK", 0), len(self.active))

    K = 3
    d = _make_driver(
        8, K, [1, 2, 3, 1, 2, 3, 1, 2], target=40, seed=5, driver_cls=_CapDriver
    )
    d.run()
    assert d._maxK <= K


def test_calibration_profiles_only_selected_client():
    server = _FakeServer(target_epochs=2)
    clients = [_CalibrationClient("C0", 0.01), _CalibrationClient("C1", 0.25)]
    profiles = {cid: ClientProfile() for cid in ("C0", "C1")}
    driver = AsyncSimDriver(
        server,
        clients,
        profiles,
        max_concurrency=1,
        logger=_silent_logger(),
        target_epochs=2,
        num_local_steps=3,
        calibration_epochs=3,
        calibration_client="C1",
    )

    driver._calibrate()

    assert clients[0].train_calls == 0
    assert clients[1].train_calls == 1
    assert driver.base_step_time == 0.25
    assert driver.timing_only is True


def test_calibration_rejects_unknown_client():
    driver = AsyncSimDriver(
        _FakeServer(target_epochs=1),
        [_CalibrationClient("C0", 0.01)],
        {"C0": ClientProfile()},
        max_concurrency=1,
        logger=_silent_logger(),
        target_epochs=1,
        num_local_steps=3,
        calibration_epochs=3,
        calibration_client="missing",
    )

    try:
        driver._calibrate()
    except ValueError as error:
        assert "Unknown calibration_client" in str(error)
    else:
        raise AssertionError("unknown calibration client should fail")
