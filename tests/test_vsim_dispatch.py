"""Tests of the MPI coordinator's task placement, with a stand-in communicator.

No MPI launch is needed: the executor talks to a fake communicator that records
what rank 0 sends and replays what workers would answer, so the tests can check
which worker gets each task, and when.
"""

import pytest

MPI = pytest.importorskip("mpi4py.MPI")
torch = pytest.importorskip("torch")

from appfl.vsim.mpi import MPIExecutor  # noqa: E402
from appfl.vsim.mpi import protocol as P  # noqa: E402

IDS = [f"C{i}" for i in range(6)]  # homes without durations: C0 -> 1, C1 -> 2, ...


class _Done:
    def Test(self):
        return True


class _FakeComm:
    """Rank 0's side of a communicator: records sends, delivers queued replies."""

    def __init__(self, workers):
        self.size = workers + 1
        self.sent = []  # (worker, control message)
        self.models = []  # (worker, payload bytes)
        self.replies = []  # (worker, control message) still to deliver

    def Get_rank(self):
        return 0

    def Get_size(self):
        return self.size

    def Dup(self):
        return self

    def Split(self, color, key):
        return None

    def isend(self, message, dest, tag):
        assert tag == P.TAG_CTRL
        self.sent.append((dest, message))
        return _Done()

    def Isend(self, buf, dest, tag):
        assert tag == P.TAG_MODEL
        self.models.append((dest, len(buf[0])))
        return _Done()

    def Irecv(self, buf, source, tag):
        return _Done()

    def Iprobe(self, source, tag, status):
        if not self.replies:
            return False
        status.Set_source(self.replies[0][0])
        return True

    def recv(self, source, tag):
        worker, message = self.replies.pop(0)
        assert source == worker
        return message


def _executor(workers=2, **kw):
    """An executor past setup: every worker holds global model version 0."""
    comm = _FakeComm(workers)
    executor = MPIExecutor(comm, IDS, **kw)
    executor._steps = {cid: 100 for cid in IDS}
    executor._last_version = {w: 0 for w in executor._workers}
    executor._model(0, {"w": torch.zeros(4)})
    return executor, comm


def _tasks(comm):
    """(worker, tid) of every TASK sent so far."""
    return [(w, m["tid"]) for w, m in comm.sent if m["kind"] == P.TASK]


def _reply(executor, comm, worker, tid, kind=P.DONE, busy=0.1):
    message = {"kind": kind, "tid": tid, "rank": worker}
    if kind == P.DONE:
        message.update(meta={}, schema=[], nbytes=0, busy=busy)
    comm.replies.append((worker, message))
    executor._progress()


def test_free_slots_take_the_earliest_waiting_task():
    executor, comm = _executor(workers=2, queue_depth=2)
    for i, deadline in enumerate([5.0, 1.0, 4.0, 2.0, 7.0, 3.0]):
        executor.submit(IDS[i], 0, None, deadline, seed=i)
    # One task per idle worker first, then a second slot each; two tasks wait.
    assert _tasks(comm) == [(1, 0), (2, 1), (1, 2), (2, 3)]
    _reply(executor, comm, 2, 1)
    assert _tasks(comm)[-1] == (2, 5)  # deadline 3.0 before 7.0
    _reply(executor, comm, 1, 0)
    assert _tasks(comm)[-1] == (1, 4)
    assert executor._stats["model_sends"] == 0  # every worker holds version 0


def test_with_one_slot_a_worker_gets_work_only_when_idle():
    executor, comm = _executor(workers=2, queue_depth=1)
    for i, deadline in enumerate([3.0, 2.0, 1.0]):
        executor.submit(IDS[i], 0, None, deadline, seed=None)
    assert _tasks(comm) == [(1, 0), (2, 1)]
    _reply(executor, comm, 2, 1)
    assert _tasks(comm) == [(1, 0), (2, 1), (2, 2)]


def test_tasks_carry_each_clients_round_count():
    executor, comm = _executor(workers=2, queue_depth=1)
    executor.submit("C0", 0, None, 1.0, seed=None)
    _reply(executor, comm, 1, 0)
    executor.submit("C0", 0, None, 2.0, seed=None)
    executor.submit("C1", 0, None, 3.0, seed=None)
    rounds = [(m["cid"], m["round"]) for _, m in comm.sent if m["kind"] == P.TASK]
    assert rounds == [("C0", 0), ("C0", 1), ("C1", 0)]


def test_tasks_go_where_they_should_finish_first():
    executor, comm = _executor(workers=2, queue_depth=2)
    # Each worker's first task also initializes CUDA, so its time is not counted;
    # then worker 1 needs 1.0 s per task and worker 2 0.1 s.
    for tid, busy in ((0, 5.0), (2, 1.0)):
        executor.submit("C0", 0, None, float(tid), seed=None)  # worker 1, its home
        executor.submit("C1", 0, None, float(tid + 1), seed=None)  # worker 2
        _reply(executor, comm, 1, tid, busy=busy)
        _reply(executor, comm, 2, tid + 1, busy=busy / 10)
    assert _tasks(comm) == [(1, 0), (2, 1), (1, 2), (2, 3)]
    executor.submit("C2", 0, None, 4.0, seed=None)  # passes over idle worker 1
    executor.submit("C3", 0, None, 5.0, seed=None)  # queues behind it on worker 2
    executor.submit("C4", 0, None, 6.0, seed=None)  # worker 2 is full
    assert _tasks(comm)[4:] == [(2, 4), (2, 5), (1, 6)]


def test_pinned_tasks_wait_for_their_home_worker():
    executor, comm = _executor(workers=2, queue_depth=1, pinned=["C1"])
    executor.submit("C0", 0, None, 1.0, seed=None)  # worker 1
    executor.submit("C2", 0, None, 2.0, seed=None)  # worker 2
    executor.submit("C1", 0, None, 0.5, seed=None)  # pinned to worker 2: waits
    executor.submit("C4", 0, None, 3.0, seed=None)  # waits
    assert _tasks(comm) == [(1, 0), (2, 1)]
    _reply(executor, comm, 1, 0)
    assert _tasks(comm)[-1] == (1, 3)  # C4: the earlier C1 may not run on worker 1
    _reply(executor, comm, 2, 1)
    assert _tasks(comm)[-1] == (2, 2)


def test_static_placement_sends_every_task_home_at_once():
    executor, comm = _executor(workers=2, placement="static", queue_depth=1)
    for i in range(6):
        executor.submit(IDS[i], 0, None, float(i), seed=None)
    assert _tasks(comm) == [(1 + i % 2, i) for i in range(6)]


def test_waiting_tasks_drop_silently_and_queued_ones_free_their_slot_on_gone():
    executor, comm = _executor(workers=1, queue_depth=1)
    first = executor.submit("C0", 0, None, 1.0, seed=None)
    executor.submit("C1", 0, None, 2.0, seed=None)
    executor.submit("C2", 0, None, 3.0, seed=None)
    executor.set_remaining(1)  # only the first task can still be consumed
    assert executor._stats["released_waiting"] == 2
    assert not [m for _, m in comm.sent if m["kind"] == P.DROP]
    executor.set_remaining(None)
    executor.submit("C3", 0, None, 0.5, seed=None)  # waits for worker 1
    executor.discard(first)
    assert [m["tid"] for _, m in comm.sent if m["kind"] == P.DROP] == [0]
    assert _tasks(comm) == [(1, 0)]  # the dropped task still holds the slot
    _reply(executor, comm, 1, 0, kind=P.GONE)
    assert _tasks(comm) == [(1, 0), (1, 3)]


def test_a_worker_gets_a_global_model_only_when_it_lacks_it():
    executor, comm = _executor(workers=2, queue_depth=2)
    v1 = {"w": torch.ones(4)}
    executor.submit("C0", 1, v1, 1.0, seed=None)  # worker 1 lacks version 1
    executor.submit("C1", 1, v1, 2.0, seed=None)  # so does worker 2
    executor.submit("C2", 1, v1, 3.0, seed=None)  # worker 1 holds it now
    executor.submit("C3", 0, {"w": torch.zeros(4)}, 4.0, seed=None)  # worker 2
    assert _tasks(comm) == [(1, 0), (2, 1), (1, 2), (2, 3)]
    assert [w for w, _ in comm.models] == [1, 2, 2]
    assert executor._stats["model_sends_saved"] == 1
    assert set(executor._models) == {1}  # only the newest stays packed


@pytest.mark.parametrize(
    "kw, match",
    [
        (dict(placement="anywhere"), "placement"),
        (dict(queue_depth=0), "queue_depth"),
        (dict(pinned=["C9"]), "unknown clients"),
    ],
)
def test_invalid_placement_settings_raise(kw, match):
    with pytest.raises(ValueError, match=match):
        MPIExecutor(_FakeComm(2), IDS, **kw)
