"""Tests for placing vsim clients on MPI workers by expected load (no MPI needed).

The async rate model is checked against the in-process driver: per-worker arrival
counts under the computed homes must be close to even.
"""

import logging
import math
import random

import pytest

from appfl.vsim import AsyncSimDriver, ClientProfile
from appfl.vsim.mpi.coordinator import _update_rates, assign_homes


class _Scheduler:
    def __init__(self):
        self.epochs = 0

    def get_num_global_epochs(self):
        return self.epochs


class _Server:
    def __init__(self, target):
        self.scheduler = _Scheduler()
        self.target = target
        self.model = None

    def get_parameters(self, **kw):
        return {}

    def global_update(self, client_id, local_model, blocking=False, **kw):
        self.scheduler.epochs += 1
        return {}

    def training_finished(self, **kw):
        return self.scheduler.epochs >= self.target


class _Client:
    def __init__(self, cid):
        self.cid = cid

    def get_id(self):
        return self.cid

    def load_parameters(self, params):
        pass

    def train(self, **kw):
        pass

    def get_parameters(self):
        return {}, {"current_local_steps": 100}


def _logger():
    lg = logging.getLogger("vsim_placement_test")
    lg.handlers.clear()
    lg.addHandler(logging.NullHandler())
    lg.setLevel(logging.CRITICAL)
    return lg


def _durations(n, seed=1):
    """Lognormal(0, 0.5) compute factors, as in the example configs."""
    rng = random.Random(seed)
    return {f"C{i}": rng.lognormvariate(0.0, 0.5) for i in range(n)}


def _arrivals_per_worker(durations, homes, workers, max_in_flight, target):
    """Run the in-process async driver and count arrivals per home worker."""
    ids = list(durations)
    driver = AsyncSimDriver(
        _Server(target),
        [_Client(cid) for cid in ids],
        {
            cid: ClientProfile(compute_factor=d, bandwidth=0)
            for cid, d in durations.items()
        },
        max_in_flight=max_in_flight,
        logger=_logger(),
        seed=3,
        base_step_time=0.01,
        task_seeding=False,  # nothing trains; skip saving and restoring the RNGs
    )
    driver.run()
    counts = [0] * workers
    for record in driver.history:
        counts[homes[record["cid"]] - 1] += 1
    return counts


def _imbalance(counts):
    return max(counts) / (sum(counts) / len(counts))


def test_without_durations_homes_are_modulo():
    ids = [f"C{i}" for i in range(7)]
    assert assign_homes(ids, 3) == {cid: 1 + i % 3 for i, cid in enumerate(ids)}


@pytest.mark.parametrize("workers", [2, 3, 5])
def test_sync_deals_clients_in_order_of_duration(workers):
    durations = _durations(17)
    homes = assign_homes(list(durations), workers, durations, mode="sync_count")
    ranked = sorted(durations, key=durations.get)
    assert [homes[cid] for cid in ranked] == [1 + i % workers for i in range(17)]
    # A count barrier accepts the fastest clients; any such set splits evenly.
    for accepted in range(1, 18):
        counts = [0] * workers
        for cid in ranked[:accepted]:
            counts[homes[cid] - 1] += 1
        assert max(counts) - min(counts) <= 1


def test_rates_follow_durations_when_every_client_is_in_flight():
    durations = _durations(12)
    rates = _update_rates(durations, None)
    assert rates == pytest.approx({cid: 1 / d for cid, d in durations.items()})
    assert _update_rates(durations, 12) == rates


def test_rates_keep_max_in_flight_clients_in_flight():
    durations = _durations(48)
    for in_flight in (1, 4, 24, 47):
        rates = _update_rates(durations, in_flight)
        # Little's law: in-flight clients = sum of rate x duration.
        assert sum(rates[c] * durations[c] for c in durations) == pytest.approx(
            in_flight
        )
        # A client's cycle is its duration plus a common idle wait.
        waits = [1 / rates[c] - durations[c] for c in durations]
        assert max(waits) - min(waits) == pytest.approx(0, abs=1e-9)


def _measured_imbalance(durations, homes, workers, max_in_flight):
    counts = _arrivals_per_worker(durations, homes, workers, max_in_flight, 4000)
    return _imbalance(counts)


@pytest.mark.parametrize("max_in_flight", [48, 24, 8])
@pytest.mark.parametrize("workers", [4, 8])
def test_async_homes_even_out_arrivals(max_in_flight, workers):
    durations = _durations(48)
    ids = list(durations)
    homes = assign_homes(ids, workers, durations, "async", max_in_flight)
    balanced = _measured_imbalance(durations, homes, workers, max_in_flight)
    assert balanced < 1.06
    if max_in_flight >= 24:
        # Fast clients make most arrivals and pile up on some modulo homes.
        modulo = assign_homes(ids, workers)
        assert (
            balanced
            < _measured_imbalance(durations, modulo, workers, max_in_flight) - 0.05
        )


@pytest.mark.parametrize("workers", [4, 8])
def test_async_rates_account_for_clients_waiting_to_be_sampled(workers):
    # With 8 of 48 clients in flight, every client mostly waits to be sampled, so
    # arrivals are nearly uniform; packing by 1 / duration alone would overload
    # the workers that hold the slow clients.
    durations = _durations(48)
    ids = list(durations)
    homes = assign_homes(ids, workers, durations, "async", 8)
    naive = assign_homes(ids, workers, durations, "async", None)
    assert _measured_imbalance(durations, naive, workers, 8) > (
        _measured_imbalance(durations, homes, workers, 8) + 0.04
    )


def test_async_homes_without_max_in_flight_pack_inverse_durations():
    durations = {"C0": 1.0, "C1": 1.0, "C2": 2.0, "C3": 2.0, "C4": 4.0, "C5": 4.0}
    homes = assign_homes(list(durations), 2, durations, "async")
    loads = [0.0, 0.0]
    for cid, worker in homes.items():
        loads[worker - 1] += 1 / durations[cid]
    assert loads == [1.75, 1.75]


@pytest.mark.parametrize(
    "durations, match",
    [
        ({"C0": 1.0}, "no entry"),
        ({"C0": 1.0, "C1": 0.0}, "positive and finite"),
        ({"C0": 1.0, "C1": -2.0}, "positive and finite"),
        ({"C0": 1.0, "C1": math.nan}, "positive and finite"),
        ({"C0": 1.0, "C1": math.inf}, "positive and finite"),
    ],
)
def test_invalid_durations_raise(durations, match):
    with pytest.raises(ValueError, match=match):
        assign_homes(["C0", "C1"], 2, durations)


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="mode"):
        assign_homes(["C0", "C1"], 2, {"C0": 1.0, "C1": 2.0}, mode="window")
