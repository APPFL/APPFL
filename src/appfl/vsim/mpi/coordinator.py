"""Coordinator side of the vsim MPI runtime: :class:`MPIExecutor`.

Rank 0 keeps the virtual clock and the APPFL server agent, and hands local training
to worker ranks through this executor. Small control messages are pickled, while
models move as raw byte buffers (see :mod:`appfl.vsim.mpi.protocol`).

The coordinator never blocks in a send. It blocks only inside :meth:`result`, while
polling for the one result the event loop needs next. Results are fetched ahead of
need in ``(deadline, tid)`` order, the order the drivers consume them in, within a
byte budget, so transfers overlap with training on the workers.

A global model handed over packed is sent from its own buffer, and
:meth:`result_packed` hands over the buffer a result arrived in. Buffers given back
through :meth:`recycle` receive later results, so the coordinator's model memory is
allocated once rather than per transfer.
"""

import bisect
import gc
import statistics
import time
from typing import Dict, List, Mapping, Optional

from ..codec import PackedState, pack_state, unpack_state
from ..executor import DeferredExecutor, TaskHandle
from . import protocol as P

_MIN_BACKOFF = 5e-5
_MAX_BACKOFF = 2e-3


def assign_homes(
    client_ids: List[str],
    num_workers: int,
    durations: Optional[Mapping[str, float]] = None,
    mode: str = "async",
    max_in_flight: Optional[int] = None,
) -> Dict[str, int]:
    """
    Choose the worker rank, ``1`` to ``num_workers``, that hosts each client.

    A worker trains every task of the clients it hosts, and every task takes about
    the same physical time whatever the client's virtual duration, so a good
    placement evens out the number of tasks each worker must train. Without
    ``durations``, client ``i`` goes to rank ``1 + i % num_workers``. With an
    expected virtual duration per client:

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
    :func:`appfl.vsim.mpi.run_worker`. Each client is homed on one worker, so its
    state never moves. Given expected durations, homes balance the expected work
    (see :func:`assign_homes`); otherwise client ``i`` in ``client_ids`` order is
    homed on rank ``1 + i % W`` for ``W`` workers.

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
        to place clients on workers. ``None`` keeps ``i % W``.
    :param mode: Driver mode the placement balances for: ``"async"``, or a sync mode.
    :param max_in_flight: async only: the driver's ``max_in_flight``.
    """

    accepts_packed = True

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
    ):
        from mpi4py import MPI

        if comm.Get_rank() != 0:
            raise ValueError("MPIExecutor must be constructed on rank 0.")
        if comm.Get_size() < 2:
            raise ValueError(
                "MPIExecutor needs at least one worker rank; launch with "
                "`mpiexec -n <1 + workers> ...`."
            )
        self._MPI = MPI
        self.comm = comm.Dup()
        # Workers split off a worker-only communicator; Split is collective.
        self.comm.Split(MPI.UNDEFINED, 0)
        self.num_workers = self.comm.Get_size() - 1
        self._workers = list(range(1, self.num_workers + 1))

        self.client_ids = list(client_ids)
        if len(set(self.client_ids)) != len(self.client_ids):
            raise ValueError("client_ids must be unique")
        self.homes = assign_homes(
            self.client_ids, self.num_workers, durations, mode, max_in_flight
        )
        if durations is None:
            self._placement = "i % W"
        elif mode.startswith("sync"):
            self._placement = "dealt in order of expected duration"
        else:
            self._placement = "packed by expected update rate"
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
        self._last_version: Dict[int, int] = {}
        self._packed = None  # (version, schema, buffer) of the newest packed model
        self._sends: List = []  # (request, buffer kept alive until completion)
        self._spare: Dict[int, List] = {}  # nbytes -> buffers free for receives
        self._spare_bytes = 0
        self._retired: List = []  # recycled buffers that may still be sending
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
            "model_sends": 0,
            "model_sends_saved": 0,
            "broadcasts": 0,
            "bytes_to_workers": 0,
            "bytes_from_workers": 0,
            "stall_seconds": 0.0,
            "pack_seconds": 0.0,
            "setup_seconds": 0.0,
            "peak_prefetched_bytes": 0,
            "buffers_allocated": 0,
            "buffers_reused": 0,
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
        schema, buf = self._pack(version, init_model)
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

        hosts = sorted({r["host"] for r in replies})
        idle = [r["rank"] for r in replies if r["clients"] == 0]
        self._info(
            f"MPI executor: {self.num_workers} workers on {len(hosts)} node(s), "
            f"{len(self.client_ids)} clients, devices: "
            f"{sorted({r['device'] for r in replies})}, torch threads per worker: "
            f"{sorted({r['threads'] for r in replies})}, client homes: "
            f"{self._placement}"
        )
        if idle:
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
        schema, buf = self._pack(version, params)
        header = {
            "kind": P.BCAST,
            "version": version,
            "schema": schema,
            "nbytes": int(buf.numel()),
        }
        for w in self._workers:
            self._send_ctrl(w, header)
        request = self.comm.Ibcast([buf.numpy(), self._MPI.BYTE], root=0)
        self._sends.append((request, buf))
        for w in self._workers:
            self._last_version[w] = version
        self._stats["broadcasts"] += 1
        self._stats["bytes_to_workers"] += int(buf.numel()) * self.num_workers

    def submit(
        self, cid: str, version: int, params, deadline: float, seed: Optional[int]
    ) -> TaskHandle:
        """
        Send a TASK to the client's home worker, with the model if it lacks it.

        A task that can never be consumed (see :meth:`set_remaining`) is not sent.
        """
        version = int(version)
        worker = self.homes[cid]
        tid = self._next_tid
        self._next_tid += 1
        handle = TaskHandle(tid, cid, version, float(deadline))
        self._stats["tasks"] += 1
        if self._beyond_horizon(handle.deadline):
            self._stats["dropped"] += 1
            self._stats["dropped_unsent"] += 1
            return handle
        message = {
            "kind": P.TASK,
            "tid": tid,
            "cid": cid,
            "version": version,
            "deadline": handle.deadline,
            "seed": seed,
        }
        if self._last_version[worker] != version:
            schema, buf = self._pack(version, params)
            message["schema"], message["nbytes"] = schema, int(buf.numel())
            self._send_ctrl(worker, message)
            request = self.comm.Isend(
                [buf.numpy(), self._MPI.BYTE], dest=worker, tag=P.TAG_MODEL
            )
            self._sends.append((request, buf))
            self._last_version[worker] = version
            self._stats["model_sends"] += 1
            self._stats["bytes_to_workers"] += int(buf.numel())
        else:
            self._send_ctrl(worker, message)
            self._stats["model_sends_saved"] += 1

        self._tasks[tid] = {"handle": handle, "worker": worker, "state": "queued"}
        bisect.insort(self._need, (handle.deadline, tid))
        self._drop_unreachable()  # the new deadline may pull the horizon in
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
        packed, meta = self.result_packed(handle)
        return unpack_state(packed.schema, packed.buffer), meta

    def result_packed(self, handle: TaskHandle):
        """
        Like :meth:`result`, but return the local model as a
        :class:`~appfl.vsim.codec.PackedState` of the buffer it arrived in. The
        caller owns that buffer, and may give it back through :meth:`recycle`.
        """
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

        packed = PackedState(record["schema"], record["buf"])
        meta = record["meta"]
        self._fetched_bytes -= record["nbytes"]
        self._stats["results"] += 1
        self._stats["bytes_from_workers"] += record["nbytes"]
        self._forget(handle.tid)
        self._prefetch_dirty = True
        self._progress()
        return packed, meta

    def recycle(self, buffer) -> None:
        """
        Take back a buffer from :meth:`result_packed` or a packed global model, to
        receive later results into once no send of it is pending.
        """
        self._retired.append(buffer)

    def discard(self, handle: TaskHandle) -> None:
        """Tell the worker to drop the task or its held result."""
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
        self.comm.Free()
        self._closed = True
        self._summary = self._summarize()
        return self._summary

    # ---------- internals ----------
    def _pack(self, version: int, params):
        """
        Return (schema, buffer) for ``version``, packing ``params`` only once.

        A :class:`~appfl.vsim.codec.PackedState` is sent from its own buffer, which
        its owner no longer writes to.
        """
        if self._packed is not None and self._packed[0] == version:
            return self._packed[1], self._packed[2]
        if params is None:
            raise RuntimeError(
                f"Global model version {version} was neither published nor supplied."
            )
        if isinstance(params, PackedState):
            schema, buf = params
        else:
            started = time.perf_counter()
            schema, buf = pack_state(params)
            self._stats["pack_seconds"] += time.perf_counter() - started
        P.check_payload_size(int(buf.numel()))
        self._packed = (version, schema, buf)
        return schema, buf

    def _take_buffer(self, nbytes: int):
        """Return a receive buffer of ``nbytes``, reusing a recycled one if free."""
        import torch

        if self._retired:
            sending = {id(keep) for _, keep in self._sends if keep is not None}
            if self._packed is not None:
                sending.add(id(self._packed[2]))  # may be sent again
            busy = []
            for buf in self._retired:
                size = int(buf.numel())
                if id(buf) in sending:
                    busy.append(buf)
                elif self._spare_bytes <= self.prefetch_bytes:
                    # Spares beyond the prefetch budget are freed instead.
                    self._spare.setdefault(size, []).append(buf)
                    self._spare_bytes += size
            self._retired = busy
        spare = self._spare.get(nbytes)
        if spare:
            self._spare_bytes -= nbytes
            self._stats["buffers_reused"] += 1
            return spare.pop()
        self._stats["buffers_allocated"] += 1
        return torch.empty(nbytes, dtype=torch.uint8)

    def _send_ctrl(self, worker: int, message: Dict) -> None:
        request = self.comm.isend(message, dest=worker, tag=P.TAG_CTRL)
        self._sends.append((request, None))

    def _fetch(self, tid: int) -> None:
        """Post the receive for a ready result, then ask its worker to send it."""
        record = self._tasks[tid]
        buf = self._take_buffer(record["nbytes"])
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
        """Forget a task: drop it on its worker, or free its payload once here."""
        tid = record["handle"].tid
        state = record["state"]
        if state in ("queued", "ready"):
            self._send_ctrl(record["worker"], {"kind": P.DROP, "tid": tid})
            self._forget(tid)
        elif state == "fetching":
            # The payload is already on its way; free it once received.
            self._remove_need(record)
            record["state"] = "discarded"
        elif state == "received":
            self._fetched_bytes -= record["nbytes"]
            self._retired.append(record["buf"])
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
                    self._retired.append(record["buf"])
                    self._tasks.pop(tid, None)
                else:
                    record["state"] = "received"
        return completed

    def _stall_report(self) -> str:
        lines = [
            f"No progress for {self.stall_timeout:.0f}s "
            f"(stall_timeout); outstanding tasks per worker:"
        ]
        per_worker: Dict[int, Dict[str, int]] = {}
        for record in self._tasks.values():
            counts = per_worker.setdefault(record["worker"], {})
            counts[record["state"]] = counts.get(record["state"], 0) + 1
        for worker in sorted(per_worker):
            lines.append(f"  rank {worker}: {per_worker[worker]}")
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
