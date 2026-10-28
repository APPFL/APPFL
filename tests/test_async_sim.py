"""Unit tests for the virtual-time async FL simulator (appfl.vsim).

These test the core event-queue logic in isolation using lightweight fakes for
ServerAgent / ClientAgent, so no dataset, model, or real training is needed.
"""

import logging
import random
from collections import Counter

import pytest

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
    # 12.5 MB over 100 Mbps: 12.5e6*8 = 100e6 bits -> 1s each way, 2s round trip.
    twelve_and_a_half_mb = 12_500_000
    assert abs(p.download_time(twelve_and_a_half_mb) - 1.0) < 1e-12
    assert abs(p.upload_time(twelve_and_a_half_mb) - 1.0) < 1e-12
    assert abs(p.comm_time(twelve_and_a_half_mb) - 2.0) < 1e-12
    assert abs(p.duration(0.01, 100, twelve_and_a_half_mb) - 4.0) < 1e-12
    assert p.next_available(123.0) == 123.0  # always available by default


def test_comm_time_is_the_two_directions():
    """comm_time composes the directional methods, so overriding either works."""

    class _AsymmetricLink(ClientProfile):
        def upload_time(self, model_bytes, **kwargs):
            return super().upload_time(model_bytes, **kwargs) * 4  # slow uplink

    p = _AsymmetricLink(bandwidth=100.0)
    mb = 12_500_000
    assert abs(p.comm_time(mb) - 5.0) < 1e-12  # 1s down + 4s up


def test_compression_ratio_scales_transfer():
    """Only the compressed bytes are put on the wire."""
    uncompressed = _make_driver(2, 2, [1.0, 1.0], target=4)
    compressed = AsyncSimDriver(
        _FakeServer(4),
        [_FakeClient(f"C{i}") for i in range(2)],
        {f"C{i}": ClientProfile(compute_factor=1.0) for i in range(2)},
        max_in_flight=2,
        logger=_silent_logger(),
        base_step_time=0.01,
        compression_ratio=0.25,
    )
    assert compressed._model_bytes == compressed._raw_model_bytes * 0.25
    assert uncompressed._model_bytes == uncompressed._raw_model_bytes


def test_invalid_compression_ratio_rejected():
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="compression_ratio"):
            AsyncSimDriver(
                _FakeServer(1),
                [_FakeClient("C0")],
                {"C0": ClientProfile()},
                max_in_flight=1,
                logger=_silent_logger(),
                compression_ratio=bad,
            )


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


class _NoTimingClient(_FakeClient):
    """Reports a step count but no per-step time, as any non-VanillaTrainer would."""

    def get_parameters(self):
        return ({}, {"current_local_steps": self.steps, "val_accuracy": self.acc})


class _NoStepsClient(_FakeClient):
    """Reports a per-step time but no step count, as "epoch" mode would."""

    def get_parameters(self):
        return ({}, {"compute_second_per_step": self.cps, "val_accuracy": self.acc})


def _driver_with(client_cls, base_step_time):
    clients = [client_cls(f"C{i}") for i in range(2)]
    profiles = {f"C{i}": ClientProfile(compute_factor=1.0) for i in range(2)}
    return AsyncSimDriver(
        _FakeServer(5),
        clients,
        profiles,
        max_in_flight=2,
        logger=_silent_logger(),
        base_step_time=base_step_time,
    )


def test_missing_step_time_raises():
    """No per-step time and no override: fail loudly, don't run at zero compute."""
    with pytest.raises(RuntimeError, match="reported compute_second_per_step=None"):
        _driver_with(_NoTimingClient, base_step_time=None).run()


def test_missing_step_count_raises():
    """base_step_time supplies seconds-per-step; a step count is still required."""
    with pytest.raises(RuntimeError, match="reported current_local_steps=None"):
        _driver_with(_NoStepsClient, base_step_time=0.01).run()


def test_base_step_time_rescues_a_trainer_without_timing():
    """The documented escape hatch: a step count plus a fixed per-step time is enough."""
    d = _driver_with(_NoTimingClient, base_step_time=0.01)
    d.run()
    assert len(d.history) == 5
    assert all(r["duration"] > 0 for r in d.history)


def test_driver_does_not_touch_the_global_rng():
    """Constructing and running a driver must not disturb the module-global RNG."""
    random.seed(1234)
    expected = [random.random() for _ in range(5)]

    random.seed(1234)
    _make_driver(4, 2, [1.0, 2.0, 0.5, 1.5], target=10, seed=99).run()
    assert [random.random() for _ in range(5)] == expected


def test_global_rng_does_not_perturb_the_driver():
    """An unrelated consumer of `random` must not shift the dispatch order."""

    def dispatch_order():
        d = _make_driver(4, 2, [1.0, 2.0, 0.5, 1.5], target=12, seed=7)
        d.run()
        return [r["cid"] for r in d.history]

    baseline = dispatch_order()
    random.seed(0)
    random.random()  # somebody else draws from the global stream
    assert dispatch_order() == baseline


def test_unavailable_client_is_deferred_not_spun():
    """A client offline until T is dispatched at T, and the clock advances there."""

    class _OfflineUntil(ClientProfile):
        """C0 cannot start before t=50; everyone else is always available."""

        def __init__(self, offline_until, **kw):
            super().__init__(**kw)
            self.offline_until = offline_until

        def next_available(self, vtime):
            return max(vtime, self.offline_until)

    # Target chosen so the run outlives t=50 and C0 really is dispatched;
    # with base_step_time=0.01 x 100 steps each round costs 1.0 virtual second.
    server = _FakeServer(70)
    clients = [_FakeClient(f"C{i}") for i in range(3)]
    profiles = {
        "C0": _OfflineUntil(50.0, compute_factor=1.0),
        "C1": ClientProfile(compute_factor=1.0),
        "C2": ClientProfile(compute_factor=1.0),
    }
    d = AsyncSimDriver(
        server,
        clients,
        profiles,
        max_in_flight=2,
        logger=_silent_logger(),
        seed=5,
        base_step_time=0.01,
    )
    d.run()  # would hang forever on a same-instant retry

    c0 = [r for r in d.history if r["cid"] == "C0"]
    assert c0, "C0 should be dispatched once its availability window opens"
    assert all(r["dispatch_time"] >= 50.0 for r in c0)
    vts = [r["vtime"] for r in d.history]
    assert vts == sorted(vts)


# ----------------------- compressed uploads ----------------------- #
_RAW_BYTES = 1000  # size of the served global model
_LINK_MBPS = 0.008  # 1000 bytes/s: the raw model takes exactly 1s each way


class _FakeTensor:
    """Just enough of a tensor for the driver to size it and move it to the CPU."""

    def __init__(self, nbytes, device="cuda"):
        self.nbytes, self.device = nbytes, device

    def numel(self):
        return self.nbytes

    def element_size(self):
        return 1

    def cpu(self):
        return _FakeTensor(self.nbytes, device="cpu")


class _SizedServer(_FakeServer):
    """Serves a model of known size and records every update it is given."""

    def __init__(self, target_epochs):
        super().__init__(target_epochs)
        self.received = []

    def get_parameters(self, **kw):
        return {"w": _FakeTensor(_RAW_BYTES)}

    def global_update(self, client_id, local_model, blocking=False, **kw):
        self.received.append((client_id, local_model))
        return super().global_update(client_id, local_model, blocking=blocking, **kw)


class _CompressingClient(_FakeClient):
    """Returns its update as bytes, as ClientAgent does with compression enabled."""

    def __init__(self, cid, payload=b"\x00" * 250):
        super().__init__(cid)
        self.payload = payload

    def get_parameters(self):
        return self.payload, super().get_parameters()[1]


class _TensorClient(_FakeClient):
    """Returns an uncompressed update whose tensor is still on the accelerator."""

    def get_parameters(self):
        return {"w": _FakeTensor(_RAW_BYTES)}, super().get_parameters()[1]


def _sized_driver(clients, compression_ratio=1.0, target=4):
    """Async driver over a `_RAW_BYTES` model, with 1s of compute per update."""
    return AsyncSimDriver(
        _SizedServer(target),
        clients,
        {c.get_id(): ClientProfile(bandwidth=_LINK_MBPS) for c in clients},
        max_in_flight=len(clients),
        logger=_silent_logger(),
        base_step_time=0.01,  # x 100 steps
        compression_ratio=compression_ratio,
    )


def test_comm_time_with_a_smaller_upload():
    """APPFL compresses only the upload, so the two legs can differ in size."""
    p = ClientProfile(bandwidth=100.0)
    mb = 12_500_000  # 1s each way at 100 Mbps
    assert abs(p.comm_time(mb, upload_bytes=mb / 4) - 1.25) < 1e-12
    assert abs(p.duration(0.01, 100, mb, upload_bytes=mb / 4) - 2.25) < 1e-12
    assert p.comm_time(mb, upload_bytes=0) == 1.0  # 0 is a size, not "unset"


def test_compressed_update_reaches_server_unchanged():
    """A compressing client hands back bytes, which the server decompresses itself."""
    clients = [_CompressingClient(f"C{i}") for i in range(3)]
    d = _sized_driver(clients, target=6)
    d.run()
    payload = {c.get_id(): c.payload for c in clients}
    assert len(d.server.received) == 6
    assert all(m is payload[cid] for cid, m in d.server.received)


@pytest.mark.parametrize("compression_ratio", [1.0, 0.25])
def test_compressed_upload_is_timed_from_its_payload(compression_ratio):
    """Download the raw model, upload the client's own payload; the ratio is moot."""
    payload = {"C0": b"\x00" * 250, "C1": b"\x00" * 500}
    d = _sized_driver(
        [_CompressingClient(cid, p) for cid, p in payload.items()],
        compression_ratio=compression_ratio,
    )
    d.run()
    for r in d.history:
        sent = len(payload[r["cid"]])
        # 1s compute + 1s for the raw model down + the payload up at 1000 bytes/s
        assert r["duration"] == pytest.approx(2.0 + sent / 1000)
        assert r["comm_bytes"] == _RAW_BYTES + sent


def test_compression_ratio_scales_the_upload_only():
    """A declared ratio models a compressor that is not run: only the uplink shrinks."""
    d = _sized_driver([_FakeClient("C0")], compression_ratio=0.25)
    d.run()
    for r in d.history:
        # 1s compute + 1s for the raw model down + 0.25s for a quarter of it up
        assert r["duration"] == pytest.approx(2.25)
        assert r["comm_bytes"] == pytest.approx(1.25 * _RAW_BYTES)


def test_uncompressed_update_still_moves_to_cpu():
    """Only bytes skip the CPU move; a tensor update reaches the server on the CPU."""
    d = _sized_driver([_TensorClient("C0")])
    d.run()
    assert d.server.received
    assert all(m["w"].device == "cpu" for _, m in d.server.received)
