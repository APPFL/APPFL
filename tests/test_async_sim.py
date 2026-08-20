"""Unit tests for the virtual-time async FL simulator (appfl.vsim).

These test the core event-queue logic in isolation using lightweight fakes for
ServerAgent / ClientAgent, so no dataset, model, or real training is needed.
"""

import logging
from collections import Counter

from appfl.vsim import AsyncSimDriver, ClientProfile


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


def _silent_logger():
    lg = logging.getLogger("vsim_test")
    lg.handlers.clear()
    lg.addHandler(logging.NullHandler())
    lg.setLevel(logging.CRITICAL)
    return lg


class _VersionTrackingServer(_FakeServer):
    """Mirrors FedAsyncAggregator's version bookkeeping.

    ``client_step`` is written on download (when ``client_id`` is supplied) and
    again on upload, and staleness is ``global_step - client_step[cid]``, exactly
    as ``FedAsyncAggregator.compute_steps`` computes it.
    """

    def __init__(self, target_epochs):
        super().__init__(target_epochs)
        self.client_step = {}
        self.observed_staleness = []

    def get_parameters(self, **kw):
        client_id = kw.get("client_id")
        if client_id is not None:
            self.client_step[client_id] = self.scheduler.epochs
        return {}

    def global_update(self, client_id, local_model, blocking=False, **kw):
        self.observed_staleness.append(
            self.scheduler.epochs - self.client_step.get(client_id, 0)
        )
        result = super().global_update(client_id, local_model, blocking=blocking, **kw)
        self.client_step[client_id] = self.scheduler.epochs
        return result


def _make_driver(
    n,
    K,
    factors,
    target=20,
    seed=42,
    base_step_time=0.01,
    driver_cls=AsyncSimDriver,
    server_cls=_FakeServer,
):
    server = server_cls(target)
    clients = [_FakeClient(f"C{i}") for i in range(n)]
    profiles = {
        f"C{i}": ClientProfile(compute_factor=factors[i], bandwidth=300.0)
        for i in range(n)
    }
    return driver_cls(
        server,
        clients,
        profiles,
        max_in_flight=K,
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


def test_in_flight_never_exceeds_limit():
    class _CapDriver(AsyncSimDriver):
        def _dispatch_idle(self):
            super()._dispatch_idle()
            self._peak = max(getattr(self, "_peak", 0), len(self.active))

    limit = 3
    d = _make_driver(
        8, limit, [1, 2, 3, 1, 2, 3, 1, 2], target=40, seed=5, driver_cls=_CapDriver
    )
    d.run()
    assert d._peak <= limit
    assert d.verify(40)["in_flight<=max_in_flight"]


def test_staleness_matches_aggregator_view():
    """Recorded staleness must equal what a version-tracking aggregator computes.

    With max_in_flight < num_clients, clients idle between dispatches and then
    receive the newest global model. Measuring staleness from a client's previous
    upload (rather than from its download) over-counts those updates.
    """
    d = _make_driver(
        6,
        2,
        [1.0, 3.0, 0.5, 2.0, 1.5, 4.0],
        target=25,
        seed=3,
        server_cls=_VersionTrackingServer,
    )
    d.run()
    recorded = [r["staleness"] for r in d.history]
    assert recorded == d.server.observed_staleness
    assert min(recorded) == 0  # a freshly dispatched client is not stale


def test_idle_client_is_not_stale():
    """With only one client in flight, nothing can land mid-training."""
    d = _make_driver(
        4,
        1,
        [1.0, 1.0, 1.0, 1.0],
        target=8,
        seed=11,
        server_cls=_VersionTrackingServer,
    )
    d.run()
    assert all(r["staleness"] == 0 for r in d.history)
    assert all(s == 0 for s in d.server.observed_staleness)
