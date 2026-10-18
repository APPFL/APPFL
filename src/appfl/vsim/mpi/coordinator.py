"""Coordinator side of the vsim MPI runtime: :class:`MPIExecutor`.

Rank 0 keeps the virtual clock and the APPFL server agent, and hands local training
to worker ranks through this executor. Small control messages are pickled, while
models move as raw byte buffers (see :mod:`appfl.vsim.mpi.protocol`).

Placement is dynamic by default. Submitted tasks wait on rank 0 in ``(deadline,
tid)`` order, and a worker with a free slot gets the earliest one, so no worker
idles while another has a backlog. A worker holds at most ``queue_depth`` tasks,
one by default: then the earliest-deadline task always goes to the next worker to
become idle. Per-task seeding makes a result independent of where its task ran.
Clients whose state must stay on one worker can be pinned to their home worker,
and ``placement="static"`` pins them all.

The coordinator never blocks in a send. It blocks only inside :meth:`result`, while
polling for the one result the event loop needs next. Results are fetched ahead of
need in ``(deadline, tid)`` order, the order the drivers consume them in, within a
byte budget, so transfers overlap with training on the workers.
"""

import bisect
import gc
import heapq
import math
import statistics
import time
from typing import Dict, Iterable, List, Mapping, Optional

from ..codec import pack_state, unpack_state
from ..executor import DeferredExecutor, TaskHandle
from . import protocol as P

_MIN_BACKOFF = 5e-5
# Rank 0 polls only while it waits for a result, and a worker may be waiting for
# it to notice a DONE and send the next task, so it polls often.
_MAX_BACKOFF = 5e-4

PLACEMENTS = ("dynamic", "static")


def assign_homes(
    client_ids: List[str],
    num_workers: int,
    durations: Optional[Mapping[str, float]] = None,
    mode: str = "async",
    max_in_flight: Optional[int] = None,
) -> Dict[str, int]:
    """
    Choose the home worker rank, ``1`` to ``num_workers``, of each client.

    A worker builds its home clients at startup and reports their step counts.
    Under static placement, and for pinned clients, the home worker also trains
    every task of the client; dynamic placement trains unpinned clients anywhere and
    uses homes only as a tie-break. Every task takes about the same physical time
    whatever the client's virtual duration, so a good static placement evens out
    the number of tasks each worker must train. Without ``durations``, client ``i``
    goes to rank ``1 + i % num_workers``. With an expected virtual duration per
    client:

    - async: a client in flight completes once per duration, so clients are
      packed by expected update rate, largest first, each onto the worker with the
      smallest total rate so far (LPT). With fewer clients in flight than clients,
      a finished client also waits until the driver samples it again. Sampling is
      uniform over idle clients, so every client waits the same mean time ``I``
      and updates at rate ``1 / (duration + I)``, where ``I`` keeps
      ``max_in_flight`` clients in flight: ``sum(d / (d + I)) == max_in_flight``.
    - sync: a barrier accepts the fastest clients of each cohort, so clients are
      dealt round-robin in order of duration, and every worker hosts an equal
      share of the clients most likely to be accepted.

    :param client_ids: Client ids in configured order.
    :param num_workers: Number of worker ranks.
    :param durations: Expected virtual duration of one task per client. Only ratios
        matter. ``None`` keeps ``i % num_workers``.
    :param mode: ``"async"``, or a sync mode (any value starting with ``"sync"``).
    :param max_in_flight: async only: clients in flight at once; ``None`` means all.
    :return: Dict mapping each client id to its worker rank.
    """
    if durations is None:
        return {cid: 1 + i % num_workers for i, cid in enumerate(client_ids)}
    missing = [cid for cid in client_ids if cid not in durations]
    if missing:
        raise ValueError(f"durations has no entry for clients {missing[:5]}")
    expected = {cid: float(durations[cid]) for cid in client_ids}
    invalid = [cid for cid, d in expected.items() if not 0 < d < float("inf")]
    if invalid:
        raise ValueError(
            f"durations must be positive and finite; got "
            f"{ {cid: expected[cid] for cid in invalid[:5]} }"
        )
    order = {cid: i for i, cid in enumerate(client_ids)}

    if mode.startswith("sync"):
        ranked = sorted(client_ids, key=lambda cid: (expected[cid], order[cid]))
        return {cid: 1 + i % num_workers for i, cid in enumerate(ranked)}
    if mode != "async":
        raise ValueError(f"mode must be 'async' or start with 'sync'; got {mode!r}")

    rates = _update_rates(expected, max_in_flight)
    loads = [0.0] * num_workers
    homes = {}
    for cid in sorted(client_ids, key=lambda cid: (-rates[cid], order[cid])):
        worker = min(range(num_workers), key=lambda j: (loads[j], j))
        homes[cid] = 1 + worker
        loads[worker] += rates[cid]
    return homes


def _update_rates(
    durations: Dict[str, float], max_in_flight: Optional[int]
) -> Dict[str, float]:
    """Return each client's expected async update rate (see :func:`assign_homes`)."""
    values = list(durations.values())
    in_flight = len(values) if max_in_flight is None else int(max_in_flight)
    in_flight = max(1, min(in_flight, len(values)))
    idle = 0.0
    if in_flight < len(values):
        # sum(d / (d + I)) falls from len(values) at I = 0 to at most sum(d) / I.
        low, high = 0.0, sum(values) / in_flight
        for _ in range(64):
            middle = 0.5 * (low + high)
            if sum(d / (d + middle) for d in values) > in_flight:
                low = middle
            else:
                high = middle
        idle = high
    return {cid: 1.0 / (d + idle) for cid, d in durations.items()}


class MPIExecutor(DeferredExecutor):
    """
    Train vsim clients on MPI worker ranks while rank 0 runs the event loop.

    Rank 0 constructs this executor and passes it to a driver; every other rank calls
    :func:`appfl.vsim.mpi.run_worker`. Every client has a home worker (see
    :func:`assign_homes`), which builds it at startup and reports its step count.
    With ``placement="dynamic"``, a task of an unpinned client trains on whichever
    worker frees up first; with ``"static"``, every task trains on its client's home,
    so client state never moves.

    :param comm: Communicator spanning the coordinator (rank 0) and the workers. It
        is duplicated, so other traffic on it cannot interfere.
    :param client_ids: Client ids in configured order.
    :param context: Picklable object passed to every worker's ``make_client``, for
        example the server's client configuration.
    :param log_dir: Directory for ``worker_<rank>.log`` files; empty keeps worker
        output on the console.
    :param calibration_runs: ``train()`` calls per worker when calibrating the step
        time; all but the last are warm-up.
    :param prefetch_mb: Budget, in MiB, for results fetched ahead of need.
    :param stall_timeout: Seconds without any progress before :meth:`result` gives
        up and reports what is outstanding on each worker.
    :param logger: Optional logger with ``info``/``warning`` (e.g. a VsimLogger).
    :param durations: Expected virtual duration of one task per client, used only
        to choose homes. ``None`` keeps ``i % W``.
    :param mode: Driver mode the homes balance for: ``"async"``, or a sync mode.
    :param max_in_flight: async only: the driver's ``max_in_flight``.
    :param placement: ``"dynamic"`` (default): the earliest-deadline waiting task
        goes to the first worker with a free slot. ``"static"``: every task trains on
        its client's home worker, as soon as it is submitted.
    :param pinned: Clients whose tasks always train on their home worker under
        dynamic placement, e.g. clients with differential-privacy accounting or
        personalized layers.
    :param queue_depth: Dynamic placement: tasks a worker holds at once, the one
        training included. ``1`` (default) keeps the deadline order global, but a
        worker idles for the few milliseconds its next task and model take to
        arrive. ``2`` hides that, which pays off for tasks of tens of milliseconds
        or very large models, but a queued task then sometimes runs before a more
        urgent one submitted after it.
    """

    def __init__(
        self,
        comm,
        client_ids: List[str],
        context=None,
        log_dir: str = "",
        calibration_runs: int = 2,
        prefetch_mb: float = 1024.0,
        stall_timeout: float = 1800.0,
        logger=None,
        durations: Optional[Mapping[str, float]] = None,
        mode: str = "async",
        max_in_flight: Optional[int] = None,
        placement: str = "dynamic",
        pinned: Optional[Iterable[str]] = None,
        queue_depth: int = 1,
    ):
        from mpi4py import MPI

        if comm.Get_rank() != 0:
            raise ValueError("MPIExecutor must be constructed on rank 0.")
        if comm.Get_size() < 2:
            raise ValueError(
                "MPIExecutor needs at least one worker rank; launch with "
                "`mpiexec -n <1 + workers> ...`."
            )
        if placement not in PLACEMENTS:
            raise ValueError(
                f"placement must be one of {PLACEMENTS}; got {placement!r}"
            )
        if int(queue_depth) < 1:
            raise ValueError(f"queue_depth must be at least 1; got {queue_depth}")
        self.client_ids = list(client_ids)
        if len(set(self.client_ids)) != len(self.client_ids):
            raise ValueError("client_ids must be unique")
        pinned = set(self.client_ids if placement == "static" else (pinned or ()))
        unknown = sorted(pinned.difference(self.client_ids))
        if unknown:
            raise ValueError(f"pinned names unknown clients {unknown[:5]}")

        self._MPI = MPI
        self.comm = comm.Dup()
        # Workers split off a worker-only communicator; Split is collective.
        self.comm.Split(MPI.UNDEFINED, 0)
        self.num_workers = self.comm.Get_size() - 1
        self._workers = list(range(1, self.num_workers + 1))

        self.homes = assign_homes(
            self.client_ids, self.num_workers, durations, mode, max_in_flight
        )
        if durations is None:
            self._home_rule = "i % W"
        elif mode.startswith("sync"):
            self._home_rule = "dealt in order of expected duration"
        else:
            self._home_rule = "packed by expected update rate"
        self.placement = placement
        self.pinned = frozenset(pinned)
        self.queue_depth = int(queue_depth)
        # Static placement sends every task to its home at once, however many.
        self._slots = math.inf if placement == "static" else self.queue_depth
        self.context = context
        self.log_dir = log_dir or ""
        self.calibration_runs = max(1, int(calibration_runs))
        self.prefetch_bytes = int(float(prefetch_mb) * 2**20)
        self.stall_timeout = float(stall_timeout)
        self.logger = logger
        self.step_time: Optional[float] = None
        self.worker_info: Dict[int, Dict] = {}

        self._steps: Dict[str, int] = {}
        self._tasks: Dict[int, Dict] = {}
        self._need: List = []  # sorted (deadline, tid) of unconsumed tasks
        self._remaining: Optional[int] = None  # results the driver may still consume
        self._fetching: Dict[int, Dict] = {}
        self._fetched_bytes = 0
        self._prefetch_dirty = False
        self._next_tid = 0
        # Placement: tasks waiting on rank 0 for a worker, as (deadline, tid) heaps
        # with lazy deletion; one shared heap, one per home for pinned clients.
        self._pool: List = []
        self._held: Dict[int, List] = {w: [] for w in self._workers}
        self._assigned: Dict[int, int] = {}  # tid -> worker, until DONE or GONE
        self._load = {w: 0 for w in self._workers}  # tasks a worker holds
        self._started_at = {w: 0.0 for w in self._workers}  # its current task began
        self._task_time = {w: 0.0 for w in self._workers}  # mean training time
        self._finished = {w: 0 for w in self._workers}  # tasks reported DONE
        self._built = {w: set() for w in self._workers}  # clients it has built
        self._rounds: Dict[str, int] = {}  # tasks submitted per client
        self._assign_dirty = False
        # Global models, packed once per version, kept while a waiting task needs
        # them; the newest stays for later submits and publishes of that version.
        self._models: Dict[int, Dict] = {}
        self._newest_model: Optional[int] = None
        self._last_version: Dict[int, int] = {}  # newest version each worker holds
        self._sends: List = []  # (request, buffer kept alive until completion)
        self._byes: Dict[int, Dict] = {}
        self._shutting_down = False
        self._closed = False
        self._summary: Dict = {}
        self._backoff = _MIN_BACKOFF
        self._last_activity = time.perf_counter()
        self._started = time.perf_counter()
        self._stats = {
            "tasks": 0,
            "results": 0,
            "discarded": 0,
            "dropped": 0,
            "dropped_unsent": 0,
            "released_waiting": 0,
            "model_sends": 0,
            "model_sends_saved": 0,
            "broadcasts": 0,
            "bytes_to_workers": 0,
            "bytes_from_workers": 0,
            "stall_seconds": 0.0,
            "pack_seconds": 0.0,
            "setup_seconds": 0.0,
            "peak_prefetched_bytes": 0,
            "peak_waiting_models": 0,
            "assigned_away": 0,
        }

    # ---------- DeferredExecutor interface ----------
    def setup(self, init_model, version: int, calibrate: bool) -> None:
        """
        Start the workers: send configuration and the initial model, then collect
        each worker's step census (and calibration time when ``calibrate``).
        """
        import torch

        MPI = self._MPI
        version = int(version)
        model = self._model(version, init_model)
        schema, buf = model["schema"], model["buf"]
        hello = {
            "kind": P.HELLO,
            "client_ids": self.client_ids,
            "homes": self.homes,
            "context": self.context,
            "log_dir": self.log_dir,
            "calibrate": bool(calibrate),
            "calibration_runs": self.calibration_runs,
            "version": version,
            "schema": schema,
            "nbytes": int(buf.numel()),
            "cudnn": {
                "deterministic": bool(torch.backends.cudnn.deterministic),
                "benchmark": bool(torch.backends.cudnn.benchmark),
            },
        }
        self.comm.bcast(hello, root=0)
        self.comm.Bcast([buf.numpy(), MPI.BYTE], root=0)
        replies = self.comm.gather(None, root=0)[1:]
        self._stats["bytes_to_workers"] += int(buf.numel()) * self.num_workers

        failures = [r for r in replies if r.get("kind") != P.READY]
        if failures:
            details = "\n".join(
                f"--- worker rank {r.get('rank')} ---\n{r.get('traceback')}"
                for r in failures
            )
            raise RuntimeError(f"Worker setup failed:\n{details}")
        for reply in replies:
            self.worker_info[reply["rank"]] = reply
            self._steps.update(reply["census"])
            self._built[reply["rank"]].update(reply["census"])
        missing = [cid for cid in self.client_ids if cid not in self._steps]
        if missing:
            raise RuntimeError(f"No worker reported a step count for {missing[:5]}")
        for w in self._workers:
            self._last_version[w] = version

        if calibrate:
            times = [r["step_time"] for r in replies if r.get("step_time")]
            if not times:
                raise RuntimeError("No worker reported a calibration step time.")
            self.step_time = float(statistics.median(times))
            self._info(
                f"calibrated step time: median {self.step_time:.6g}s "
                f"(min {min(times):.6g}s, max {max(times):.6g}s, "
                f"{len(times)} workers); pin it with --base_step_time "
                f"{self.step_time:.6g} to reproduce this schedule"
            )
            if min(times) > 0 and max(times) / min(times) > 1.2:
                self._warning(
                    "calibration times differ by more than 20% across workers; "
                    "the hardware may be heterogeneous or shared"
                )

        # APPFL calls gc.collect() on every update and download (optimize_memory).
        # Freezing the long-lived setup objects keeps those collections cheap; with
        # the server and torch loaded, each full collection otherwise costs tenths
        # of a second and dominates the coordinator's loop.
        gc.collect()
        gc.freeze()

        self._stats["setup_seconds"] = time.perf_counter() - self._started
        self._started = time.perf_counter()  # wall and utilization cover the run
        for w in self._workers:
            self._started_at[w] = self._started

        hosts = sorted({r["host"] for r in replies})
        if self.placement == "static":
            placement = f"static (homes {self._home_rule})"
        else:
            placement = (
                f"dynamic, {self.queue_depth} task(s) per worker, "
                f"{len(self.pinned)} pinned client(s) (homes {self._home_rule})"
            )
        self._info(
            f"MPI executor: {self.num_workers} workers on {len(hosts)} node(s), "
            f"{len(self.client_ids)} clients, devices: "
            f"{sorted({r['device'] for r in replies})}, torch threads per worker: "
            f"{sorted({r['threads'] for r in replies})}, placement: {placement}"
        )
        idle = [r["rank"] for r in replies if r["clients"] == 0]
        if idle and self.placement == "static":
            self._warning(
                f"{len(idle)} worker(s) host no client (num_clients < workers): "
                f"ranks {idle[:8]}"
            )

    def expected_steps(self, cid: str) -> int:
        """Return the census step count for ``cid``."""
        return self._steps[cid]

    def publish(self, version: int, params) -> None:
        """Broadcast global model ``version`` to every worker, once per version."""
        version = int(version)
        if all(self._last_version[w] == version for w in self._workers):
            return
        model = self._model(version, params)
        header = {
            "kind": P.BCAST,
            "version": version,
            "schema": model["schema"],
            "nbytes": int(model["buf"].numel()),
        }
        for w in self._workers:
            self._send_ctrl(w, header)
        request = self.comm.Ibcast([model["buf"].numpy(), self._MPI.BYTE], root=0)
        self._sends.append((request, model["buf"]))
        for w in self._workers:
            self._last_version[w] = max(self._last_version[w], version)
        self._stats["broadcasts"] += 1
        self._stats["bytes_to_workers"] += int(model["buf"].numel()) * self.num_workers

    def submit(
        self, cid: str, version: int, params, deadline: float, seed: Optional[int]
    ) -> TaskHandle:
        """
        Queue a task for the next free worker (its client's home, if pinned).

        A task that can never be consumed (see :meth:`set_remaining`) is not sent.
        """
        version = int(version)
        tid = self._next_tid
        self._next_tid += 1
        handle = TaskHandle(tid, cid, version, float(deadline))
        self._stats["tasks"] += 1
        if self._beyond_horizon(handle.deadline):
            self._stats["dropped"] += 1
            self._stats["dropped_unsent"] += 1
            return handle
        model = self._model(version, params)
        model["waiting"] += 1
        rounds = self._rounds.get(cid, 0)
        self._rounds[cid] = rounds + 1
        self._tasks[tid] = {
            "handle": handle,
            "worker": None,
            "state": "waiting",
            "seed": seed,
            "round": rounds,
            "model": model,
        }
        key = (handle.deadline, tid)
        if cid in self.pinned:
            heapq.heappush(self._held[self.homes[cid]], key)
        else:
            heapq.heappush(self._pool, key)
        bisect.insort(self._need, key)
        self._drop_unreachable()  # the new deadline may pull the horizon in
        self._assign_dirty = True
        self._stats["peak_waiting_models"] = max(
            self._stats["peak_waiting_models"], len(self._models)
        )
        self._progress()
        return handle

    def set_remaining(self, count: Optional[int]) -> None:
        """
        Drop every task past the ``count``-th smallest pending deadline.

        The driver consumes at most ``count`` more results, in deadline order, so
        those tasks can never be consumed. Tasks submitted later are checked too.
        """
        self._remaining = None if count is None else max(0, int(count))
        self._drop_unreachable()

    def result(self, handle: TaskHandle):
        """Block until ``handle``'s local model is here; return (model, meta)."""
        record = self._tasks.get(handle.tid)
        if record is None:
            raise RuntimeError(
                f"Task {handle.tid} is unknown, already consumed, or was dropped "
                f"because the driver declared it would never consume it"
            )
        started = time.perf_counter()
        self._last_activity = started
        while record["state"] != "received":
            if record["state"] == "ready":
                self._fetch(handle.tid)  # on demand, even beyond the budget
            self._progress(block=True)
        self._stats["stall_seconds"] += time.perf_counter() - started

        model = unpack_state(record["schema"], record["buf"])
        meta = record["meta"]
        self._fetched_bytes -= record["nbytes"]
        self._stats["results"] += 1
        self._stats["bytes_from_workers"] += record["nbytes"]
        self._forget(handle.tid)
        self._prefetch_dirty = True
        self._progress()
        return model, meta

    def discard(self, handle: TaskHandle) -> None:
        """Forget the task: unqueue it, or tell its worker to drop it or its result."""
        record = self._tasks.get(handle.tid)
        if record is None or record["state"] == "discarded":
            return
        self._stats["discarded"] += 1
        self._release(record)

    def shutdown(self) -> Dict:
        """Stop the workers, drain every transfer, free the communicator."""
        if self._closed:
            return self._summary
        self._shutting_down = True
        for w in self._workers:
            self._send_ctrl(w, {"kind": P.SHUTDOWN})
        self._last_activity = time.perf_counter()
        while len(self._byes) < self.num_workers:
            self._progress(block=True)
        pending = [request for request, _ in self._sends]
        pending += [record["req"] for record in self._fetching.values()]
        self._MPI.Request.Waitall(pending)
        self._sends, self._fetching = [], {}
        self._models.clear()
        self.comm.Free()
        self._closed = True
        self._summary = self._summarize()
        return self._summary

    # ---------- internals ----------
    def _model(self, version: int, params) -> Dict:
        """Return the packed global model ``version``, packing ``params`` once."""
        entry = self._models.get(version)
        if entry is not None:
            return entry
        if params is None:
            raise RuntimeError(
                f"Global model version {version} was neither published nor supplied."
            )
        started = time.perf_counter()
        schema, buf = pack_state(params)
        self._stats["pack_seconds"] += time.perf_counter() - started
        P.check_payload_size(int(buf.numel()))
        entry = {"version": version, "schema": schema, "buf": buf, "waiting": 0}
        self._models[version] = entry
        if self._newest_model is None or version > self._newest_model:
            previous = self._newest_model
            self._newest_model = version
            if previous is not None and not self._models[previous]["waiting"]:
                del self._models[previous]
        return entry

    def _unref_model(self, entry: Dict) -> None:
        """A waiting task no longer needs ``entry``; free it unless it is the newest."""
        entry["waiting"] -= 1
        version = entry["version"]
        if not entry["waiting"] and version != self._newest_model:
            self._models.pop(version, None)

    def _send_ctrl(self, worker: int, message: Dict) -> None:
        request = self.comm.isend(message, dest=worker, tag=P.TAG_CTRL)
        self._sends.append((request, None))

    def _assign(self) -> None:
        """Give waiting tasks to workers with free slots, earliest deadline first."""
        self._assign_dirty = False
        if self._shutting_down:
            return
        while True:
            free = [w for w in self._workers if self._load[w] < self._slots]
            if not free:
                return
            best, home = self._peek(self._pool), None
            for w in free:
                key = self._peek(self._held[w])
                if key is not None and (best is None or key < best):
                    best, home = key, w
            if best is None:
                return
            if home is None:
                heapq.heappop(self._pool)
                worker = self._choose_worker(free, self._tasks[best[1]])
            else:
                heapq.heappop(self._held[home])
                worker = home
            self._send_task(worker, self._tasks[best[1]])

    def _peek(self, heap: List):
        """Return the earliest live (deadline, tid) of ``heap``, or ``None``."""
        while heap:
            record = self._tasks.get(heap[0][1])
            if record is not None and record["state"] == "waiting":
                return heap[0]
            heapq.heappop(heap)  # released while waiting
        return None

    def _choose_worker(self, free: List[int], record: Dict) -> int:
        """
        Pick a worker for ``record`` among those with a free slot.

        The task goes where it should finish first, from each worker's mean
        training time and, for a busy worker, when its current task started. Every
        worker within a tenth of a task of the soonest qualifies, so on uniform
        hardware an idle worker wins and timing noise decides nothing. Among those,
        the idle ones come first, then a worker that already holds the task's
        global model (no transfer), then one that has built the client, then the
        client's home.
        """
        handle = record["handle"]
        home = self.homes[handle.cid]
        now = time.perf_counter()
        finish = {}
        for w in free:
            start = now
            if self._load[w]:
                held = self._load[w] * self._task_time[w]
                start = max(now, self._started_at[w] + held)
            finish[w] = start + self._task_time[w]
        soonest = min(free, key=finish.get)
        cutoff = finish[soonest] + 0.1 * self._task_time[soonest]

        def rank(w):
            return (
                self._load[w],
                self._last_version[w] != handle.version,
                handle.cid not in self._built[w],
                w != home,
                w,
            )

        return min((w for w in free if finish[w] <= cutoff), key=rank)

    def _send_task(self, worker: int, record: Dict) -> None:
        """Send a waiting task to ``worker``, with its global model if needed."""
        handle = record["handle"]
        model = record.pop("model")
        message = {
            "kind": P.TASK,
            "tid": handle.tid,
            "cid": handle.cid,
            "version": handle.version,
            "deadline": handle.deadline,
            "seed": record["seed"],
            "round": record["round"],
        }
        # A worker always keeps the newest version it received.
        if self._last_version[worker] != handle.version:
            buf = model["buf"]
            message["schema"], message["nbytes"] = model["schema"], int(buf.numel())
            self._send_ctrl(worker, message)
            request = self.comm.Isend(
                [buf.numpy(), self._MPI.BYTE], dest=worker, tag=P.TAG_MODEL
            )
            self._sends.append((request, buf))
            self._last_version[worker] = max(self._last_version[worker], handle.version)
            self._stats["model_sends"] += 1
            self._stats["bytes_to_workers"] += int(buf.numel())
        else:
            self._send_ctrl(worker, message)
            self._stats["model_sends_saved"] += 1
        self._unref_model(model)
        record.update(worker=worker, state="queued")
        if self._load[worker] == 0:
            self._started_at[worker] = time.perf_counter()
        self._load[worker] += 1
        self._assigned[handle.tid] = worker
        self._built[worker].add(handle.cid)
        if worker != self.homes[handle.cid]:
            self._stats["assigned_away"] += 1

    def _slot_freed(self, tid: int) -> None:
        """A worker answered ``tid`` with DONE or GONE: it holds one task fewer."""
        worker = self._assigned.pop(tid, None)
        if worker is None:
            return
        self._load[worker] -= 1
        if self._load[worker]:
            self._started_at[worker] = time.perf_counter()  # its next task began
        self._assign_dirty = True

    def _fetch(self, tid: int) -> None:
        """Post the receive for a ready result, then ask its worker to send it."""
        import torch

        record = self._tasks[tid]
        buf = torch.empty(record["nbytes"], dtype=torch.uint8)
        record["req"] = self.comm.Irecv(
            [buf.numpy(), self._MPI.BYTE], source=record["worker"], tag=P.TAG_RESULT
        )
        record["buf"] = buf
        record["state"] = "fetching"
        self._fetching[tid] = record
        self._send_ctrl(record["worker"], {"kind": P.FETCH, "tid": tid})
        self._fetched_bytes += record["nbytes"]
        self._stats["peak_prefetched_bytes"] = max(
            self._stats["peak_prefetched_bytes"], self._fetched_bytes
        )

    def _prefetch(self) -> None:
        """Fetch ready results in need order while the byte budget allows."""
        self._prefetch_dirty = False
        if self._shutting_down:
            return
        budget = self.prefetch_bytes - self._fetched_bytes
        for _, tid in self._need:
            if budget <= 0:
                break
            record = self._tasks[tid]
            if record["state"] == "ready" and record["nbytes"] <= budget:
                self._fetch(tid)
                budget -= record["nbytes"]

    def _release(self, record: Dict) -> None:
        """Forget a task: unqueue it, drop it on its worker, or free its payload."""
        tid = record["handle"].tid
        state = record["state"]
        if state == "waiting":
            self._stats["released_waiting"] += 1
            self._unref_model(record.pop("model"))
            self._forget(tid)  # its heap entry is skipped from now on
        elif state in ("queued", "ready"):
            self._send_ctrl(record["worker"], {"kind": P.DROP, "tid": tid})
            self._forget(tid)
        elif state == "fetching":
            # The payload is already on its way; free it once received.
            self._remove_need(record)
            record["state"] = "discarded"
        elif state == "received":
            self._fetched_bytes -= record["nbytes"]
            self._forget(tid)

    def _beyond_horizon(self, deadline: float) -> bool:
        """Whether a task due at ``deadline``, if submitted now, is never consumed."""
        remaining = self._remaining
        if remaining is None:
            return False
        if remaining == 0:
            return True
        return len(self._need) >= remaining and deadline > self._need[remaining - 1][0]

    def _drop_unreachable(self) -> None:
        """
        Drop pending tasks due after the ``r``-th smallest pending deadline.

        With ``r`` results left to consume, the driver has consumed them all by that
        deadline, whatever it submits later: later tasks can only add arrivals. Ties
        with that deadline are kept.
        """
        remaining = self._remaining
        if remaining is None or len(self._need) <= remaining:
            return
        cut = 0
        if remaining > 0:
            horizon = self._need[remaining - 1][0]
            cut = bisect.bisect_right(self._need, (horizon, float("inf")))
        unreachable = self._need[cut:]
        for _, tid in unreachable:
            self._stats["dropped"] += 1
            self._release(self._tasks[tid])
        if unreachable:
            self._prefetch_dirty = True  # dropped prefetches free budget

    def _remove_need(self, record: Dict) -> None:
        handle = record["handle"]
        key = (handle.deadline, handle.tid)
        index = bisect.bisect_left(self._need, key)
        if index < len(self._need) and self._need[index] == key:
            del self._need[index]

    def _forget(self, tid: int) -> None:
        record = self._tasks.pop(tid)
        self._remove_need(record)
        self._fetching.pop(tid, None)

    def _progress(self, block: bool = False) -> bool:
        """Handle incoming control messages and finished transfers."""
        active = self._drain_ctrl()
        active = self._test_requests() or active
        if self._assign_dirty:
            self._assign()
        if self._prefetch_dirty:
            self._prefetch()
        now = time.perf_counter()
        if active:
            self._last_activity = now
            self._backoff = _MIN_BACKOFF
        elif block:
            if now - self._last_activity > self.stall_timeout:
                raise RuntimeError(self._stall_report())
            time.sleep(self._backoff)
            self._backoff = min(self._backoff * 2, _MAX_BACKOFF)
        return active

    def _drain_ctrl(self) -> bool:
        MPI = self._MPI
        status = MPI.Status()
        received = False
        while self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=P.TAG_CTRL, status=status):
            source = status.Get_source()
            message = self.comm.recv(source=source, tag=P.TAG_CTRL)
            received = True
            kind = message["kind"]
            if kind == P.DONE:
                self._on_done(message)
            elif kind == P.GONE:
                self._slot_freed(message["tid"])
            elif kind == P.BYE:
                self._byes[source] = message["stats"]
            elif kind == P.ERROR:
                raise RuntimeError(
                    f"Worker rank {source} failed on task {message.get('tid')}:\n"
                    f"{message.get('traceback')}"
                )
            else:
                raise RuntimeError(f"Unexpected {kind!r} message from rank {source}")
        return received

    def _on_done(self, message: Dict) -> None:
        worker = self._assigned.get(message["tid"])
        if worker is not None and message.get("busy") is not None:
            # A worker's first task also initializes CUDA and cuDNN; skip it.
            self._finished[worker] += 1
            if self._finished[worker] > 1:
                seen = self._task_time[worker]
                busy = float(message["busy"])
                self._task_time[worker] = 0.8 * seen + 0.2 * busy if seen else busy
        self._slot_freed(message["tid"])
        record = self._tasks.get(message["tid"])
        if record is None or record["state"] != "queued" or self._shutting_down:
            return  # dropped or discarded in the meantime
        record.update(
            state="ready",
            meta=message["meta"],
            schema=message["schema"],
            nbytes=int(message["nbytes"]),
        )
        self._prefetch_dirty = True

    def _test_requests(self) -> bool:
        completed = False
        if self._sends:
            remaining = []
            for request, keep in self._sends:
                if request.Test():
                    completed = True
                else:
                    remaining.append((request, keep))
            self._sends = remaining
        for tid in list(self._fetching):
            record = self._fetching[tid]
            if record["req"].Test():
                completed = True
                del self._fetching[tid]
                record["req"] = None
                if record["state"] == "discarded":
                    self._fetched_bytes -= record["nbytes"]
                    self._tasks.pop(tid, None)
                else:
                    record["state"] = "received"
        return completed

    def _stall_report(self) -> str:
        lines = [
            f"No progress for {self.stall_timeout:.0f}s "
            f"(stall_timeout); outstanding tasks per worker:"
        ]
        per_worker: Dict = {}
        for record in self._tasks.values():
            counts = per_worker.setdefault(record["worker"] or 0, {})
            counts[record["state"]] = counts.get(record["state"], 0) + 1
        for worker in sorted(per_worker):
            where = f"rank {worker}" if worker else "waiting on rank 0"
            lines.append(f"  {where}: {per_worker[worker]}")
        lines.append(f"  tasks held per worker: {self._load}")
        if self._need:
            deadline, tid = self._need[0]
            record = self._tasks[tid]
            lines.append(
                f"  next needed: task {tid} ({record['handle'].cid}, deadline "
                f"{deadline:.4f}) on rank {record['worker']}, state {record['state']}"
            )
        return "\n".join(lines)

    def _summarize(self) -> Dict:
        wall = time.perf_counter() - self._started
        busy = sum(stats.get("busy_seconds", 0.0) for stats in self._byes.values())
        summary = dict(self._stats)
        summary.update(
            placement=self.placement,
            queue_depth=self.queue_depth,
            pinned=len(self.pinned),
            trained=sum(stats.get("tasks", 0) for stats in self._byes.values()),
            workers=self.num_workers,
            wall_seconds=wall,
            worker_busy_seconds=busy,
            worker_utilization=busy / (wall * self.num_workers) if wall > 0 else 0.0,
            worker_stats={rank: self._byes[rank] for rank in sorted(self._byes)},
        )
        return summary

    def _info(self, text: str) -> None:
        if self.logger is not None:
            self.logger.info(text)

    def _warning(self, text: str) -> None:
        if self.logger is not None:
            self.logger.warning(text)
