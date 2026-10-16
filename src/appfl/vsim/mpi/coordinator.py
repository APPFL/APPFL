"""Coordinator side of the vsim MPI runtime: :class:`MPIExecutor`.

Rank 0 keeps the virtual clock and the APPFL server agent, and hands local training
to worker ranks through this executor. Small control messages are pickled, while
models move as raw byte buffers (see :mod:`appfl.vsim.mpi.protocol`).

The coordinator never blocks in a send. It blocks only inside :meth:`result`, while
polling for the one result the event loop needs next. Results are fetched ahead of
need in ``(deadline, tid)`` order, the order the drivers consume them in, within a
byte budget, so transfers overlap with training on the workers.
"""

import bisect
import gc
import statistics
import time
from typing import Dict, List, Optional

from ..codec import pack_state, unpack_state
from ..executor import DeferredExecutor, TaskHandle
from . import protocol as P

_MIN_BACKOFF = 5e-5
_MAX_BACKOFF = 2e-3


class MPIExecutor(DeferredExecutor):
    """
    Train vsim clients on MPI worker ranks while rank 0 runs the event loop.

    Rank 0 constructs this executor and passes it to a driver; every other rank calls
    :func:`appfl.vsim.mpi.run_worker`. Client ``i`` in ``client_ids`` order is homed
    on rank ``1 + i % W`` for ``W`` workers, so a client's state never moves and
    every cohort spreads evenly.

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
        self.homes = {
            cid: 1 + i % self.num_workers for i, cid in enumerate(self.client_ids)
        }
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
        self._fetching: Dict[int, Dict] = {}
        self._fetched_bytes = 0
        self._prefetch_dirty = False
        self._next_tid = 0
        self._last_version: Dict[int, int] = {}
        self._packed = None  # (version, schema, buffer) of the newest packed model
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
            "model_sends": 0,
            "model_sends_saved": 0,
            "broadcasts": 0,
            "bytes_to_workers": 0,
            "bytes_from_workers": 0,
            "stall_seconds": 0.0,
            "pack_seconds": 0.0,
            "setup_seconds": 0.0,
            "peak_prefetched_bytes": 0,
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
            f"{sorted({r['device'] for r in replies})}"
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
        """Send a TASK to the client's home worker, with the model if it lacks it."""
        version = int(version)
        worker = self.homes[cid]
        tid = self._next_tid
        self._next_tid += 1
        handle = TaskHandle(tid, cid, version, float(deadline))
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
        self._stats["tasks"] += 1
        self._progress()
        return handle

    def result(self, handle: TaskHandle):
        """Block until ``handle``'s local model is here; return (model, meta)."""
        record = self._tasks.get(handle.tid)
        if record is None:
            raise RuntimeError(f"Task {handle.tid} is unknown or already consumed")
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
        """Tell the worker to drop the task or its held result."""
        record = self._tasks.get(handle.tid)
        if record is None:
            return
        self._stats["discarded"] += 1
        state = record["state"]
        if state in ("queued", "ready"):
            self._send_ctrl(record["worker"], {"kind": P.DROP, "tid": handle.tid})
            self._forget(handle.tid)
        elif state == "fetching":
            # The payload is already on its way; free it once received.
            self._remove_need(record)
            record["state"] = "discarded"
        elif state == "received":
            self._fetched_bytes -= record["nbytes"]
            self._forget(handle.tid)

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
        """Return (schema, buffer) for ``version``, packing ``params`` only once."""
        if self._packed is not None and self._packed[0] == version:
            return self._packed[1], self._packed[2]
        if params is None:
            raise RuntimeError(
                f"Global model version {version} was neither published nor supplied."
            )
        started = time.perf_counter()
        schema, buf = pack_state(params)
        self._stats["pack_seconds"] += time.perf_counter() - started
        P.check_payload_size(int(buf.numel()))
        self._packed = (version, schema, buf)
        return schema, buf

    def _send_ctrl(self, worker: int, message: Dict) -> None:
        request = self.comm.isend(message, dest=worker, tag=P.TAG_CTRL)
        self._sends.append((request, None))

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
