"""Worker side of the vsim MPI runtime: :func:`run_worker`.

Each worker rank hosts its share of the virtual clients and trains them on one
device. The main thread makes every MPI call (receiving tasks and models, joining
broadcasts, sending results on request), while a training thread runs local training
in ``(deadline, tid)`` order. The task the coordinator is waiting for therefore
trains next, and transfers overlap with training.
"""

import gc
import heapq
import os
import queue
import sys
import threading
import time
import traceback
from typing import Callable, Dict, Optional

from ..codec import pack_state, unpack_state
from ..seeding import rng_guard, task_seed
from . import protocol as P

_MIN_BACKOFF = 5e-5
_MAX_BACKOFF = 2e-3


def expected_local_steps(client) -> int:
    """
    Return how many local steps the client's next ``train()`` will run.

    A client may provide ``expected_local_steps()`` itself; otherwise the count is
    read from a VanillaTrainer-style trainer: ``num_local_steps`` in step mode, and
    ``num_local_epochs * len(train_dataloader)`` in epoch mode.

    :param client: A ClientAgent-like object.
    :return: Number of local steps.
    """
    hook = getattr(client, "expected_local_steps", None)
    if callable(hook):
        return int(hook())
    trainer = getattr(client, "trainer", None)
    configs = getattr(trainer, "train_configs", None)
    mode = configs.get("mode", None) if configs is not None else None
    if mode == "step":
        return int(configs.get("num_local_steps"))
    if mode == "epoch":
        return int(configs.get("num_local_epochs")) * len(trainer.train_dataloader)
    raise RuntimeError(
        f"Cannot tell how many local steps client {client.get_id()} will run: its "
        f"trainer exposes no `train_configs.mode` of 'step' or 'epoch'. Distributed "
        f"simulation fixes every duration at dispatch, so the step count must be "
        f"known before training; give the client an `expected_local_steps()` method."
    )


def run_worker(
    comm,
    make_client: Callable,
    device: str = "cpu",
    offload: bool = True,
) -> Dict:
    """
    Serve local-training tasks from the coordinator on rank 0 until it shuts down.

    Call it on every rank except 0, with the same communicator rank 0 passes to its
    :class:`~appfl.vsim.mpi.MPIExecutor`.

    :param comm: Communicator spanning the coordinator and all workers.
    :param make_client: ``make_client(cid, context, device)`` returning a
        ready-to-train ClientAgent-like object (configuration already loaded).
        ``context`` is what the coordinator passed to the executor.
    :param device: ``"cpu"`` or ``"cuda"``. With CUDA, the workers of a node are
        mapped to its visible GPUs round-robin. On CPU, a worker uses one torch
        thread unless ``OMP_NUM_THREADS`` is set, so the ranks sharing a node do
        not oversubscribe its cores.
    :param offload: Move a client's model back to the CPU after each task, so GPU
        memory holds one model at a time however many clients a worker hosts.
    :return: This worker's statistics (also reported to the coordinator).
    """
    worker = _Worker(comm, make_client, device, offload)
    try:
        worker.setup()
        worker.serve()
    except BaseException:
        sys.stderr.write(
            f"vsim MPI worker rank {worker.rank} failed:\n{traceback.format_exc()}"
        )
        sys.stderr.flush()
        worker.comm.Abort(1)
        raise
    return worker.stats


class _Worker:
    """State and loops of one worker rank."""

    def __init__(self, comm, make_client, device, offload):
        from mpi4py import MPI

        self.MPI = MPI
        self.comm = comm.Dup()
        self.rank = self.comm.Get_rank()
        # Collective with the coordinator's `Split(MPI.UNDEFINED, 0)`.
        self.worker_comm = self.comm.Split(0, self.rank)
        self.node_comm = self.worker_comm.Split_type(MPI.COMM_TYPE_SHARED)
        self.make_client = make_client
        self.requested_device = device or "cpu"
        self.device = "cpu"
        self.offload = offload
        self.clients: Dict = {}

        # Shared with the training thread, guarded by `cv`.
        self.cv = threading.Condition()
        self.heap = []  # (deadline, tid, task)
        self.versions: Dict[int, Dict] = {}  # version -> schema, buffer, ready
        self.refs: Dict[int, int] = {}  # version -> unstarted tasks using it
        self.newest = -1
        self.running: Optional[int] = None
        self.dropped_running = set()
        self.stopping = False

        self.outbox = queue.Queue()
        self.results: Dict[int, object] = {}  # tid -> packed local model
        self.version_requests = []  # (version, request, buffer)
        self.sends = []  # (request, buffer)
        self.thread = None
        self.stats = {
            "rank": self.rank,
            "host": MPI.Get_processor_name(),
            "device": "cpu",
            "clients": 0,
            "tasks": 0,
            "dropped": 0,
            "busy_seconds": 0.0,
            "bytes_in": 0,
            "bytes_out": 0,
        }

    # ---------- setup ----------
    def setup(self) -> None:
        import torch

        MPI = self.MPI
        hello = self.comm.bcast(None, root=0)
        self._redirect_output(hello["log_dir"])
        v0 = torch.empty(hello["nbytes"], dtype=torch.uint8)
        self.comm.Bcast([v0.numpy(), MPI.BYTE], root=0)
        self._add_version(hello["version"], hello["schema"], v0, ready=True)
        self.stats["bytes_in"] += int(v0.numel())

        failure = None
        census, step_time = {}, None
        try:
            torch.backends.cudnn.deterministic = hello["cudnn"]["deterministic"]
            torch.backends.cudnn.benchmark = hello["cudnn"]["benchmark"]
            self.device = self._pick_device()
            if self.device.startswith("cuda"):
                torch.cuda.set_device(self.device)
            elif self.device == "cpu" and "OMP_NUM_THREADS" not in os.environ:
                # torch would start one thread per core in every rank; with 8
                # workers on one node that made a run 1.85x slower.
                torch.set_num_threads(1)
            homes, context = hello["homes"], hello["context"]
            for cid in hello["client_ids"]:
                if homes[cid] == self.rank:
                    self.clients[cid] = self.make_client(cid, context, self.device)
            census = {cid: expected_local_steps(c) for cid, c in self.clients.items()}
        except Exception:
            failure = traceback.format_exc()

        if hello["calibrate"]:
            # Workers sharing a node take turns, so each measurement is uncontended.
            for turn in range(self.node_comm.Get_size()):
                if turn == self.node_comm.Get_rank() and failure is None:
                    try:
                        step_time = self._calibrate(hello)
                    except Exception:
                        failure = traceback.format_exc()
                self.node_comm.Barrier()

        self.stats.update(device=self.device, clients=len(self.clients))
        if failure is None:
            reply = {
                "kind": P.READY,
                "rank": self.rank,
                "host": self.stats["host"],
                "device": self.device,
                "threads": torch.get_num_threads(),
                "clients": len(self.clients),
                "census": census,
                "step_time": step_time,
            }
        else:
            reply = {"kind": P.ERROR, "rank": self.rank, "traceback": failure}
        # On failure the coordinator raises and aborts the job; keep serving until
        # then, so the coordinator gets to report this traceback first.
        self.comm.gather(reply, root=0)
        gc.collect()
        gc.freeze()  # APPFL calls gc.collect() often; skip the long-lived objects

    def _pick_device(self) -> str:
        requested = str(self.requested_device)
        if not requested.startswith("cuda"):
            return requested
        import torch

        count = torch.cuda.device_count()
        if count == 0:
            raise RuntimeError(
                f"device={requested!r} but worker rank {self.rank} sees no GPU"
            )
        if ":" in requested:
            return requested
        return f"cuda:{self.node_comm.Get_rank() % count}"

    def _calibrate(self, hello) -> float:
        """Time local training on a throwaway client; return seconds per step."""
        import torch

        home = [cid for cid in hello["client_ids"] if hello["homes"][cid] == self.rank]
        cid = home[0] if home else hello["client_ids"][0]
        client = self.make_client(cid, hello["context"], self.device)
        entry = self.versions[hello["version"]]
        state = unpack_state(entry["schema"], entry["buf"])
        step_time = None
        with rng_guard(task_seed(0, f"calibration:{cid}", self.rank)):
            for _ in range(int(hello["calibration_runs"])):
                client.load_parameters(state)
                client.train()
                result = client.get_parameters()
                meta = result[1] if isinstance(result, tuple) else {}
                step_time = meta.get("compute_second_per_step")
        del client, state
        gc.collect()
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        if step_time is None or float(step_time) <= 0:
            raise RuntimeError(
                f"Client {cid} reported compute_second_per_step={step_time!r}, so "
                f"the step time cannot be calibrated. Set `base_step_time` under "
                f"`server_configs.simulator` instead."
            )
        return float(step_time)

    def _redirect_output(self, log_dir: str) -> None:
        if not log_dir:
            return
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, f"worker_{self.rank}.log")
        sys.stdout.flush()
        sys.stderr.flush()
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        os.close(fd)

    # ---------- main loop ----------
    def serve(self) -> None:
        MPI = self.MPI
        self.thread = threading.Thread(
            target=self._train_loop, name="vsim-trainer", daemon=True
        )
        self.thread.start()
        status = MPI.Status()
        backoff = _MIN_BACKOFF
        while True:
            active = False
            while self.comm.Iprobe(source=0, tag=P.TAG_CTRL, status=status):
                message = self.comm.recv(source=0, tag=P.TAG_CTRL)
                active = True
                if message["kind"] == P.SHUTDOWN:
                    self._shutdown()
                    return
                self._handle(message)
            active = self._drain_outbox() or active
            active = self._test_requests() or active
            if active:
                backoff = _MIN_BACKOFF
            else:
                time.sleep(backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF)

    def _handle(self, message: Dict) -> None:
        import torch

        kind = message["kind"]
        if kind == P.TASK:
            version = message["version"]
            if "schema" in message:
                buf = torch.empty(message["nbytes"], dtype=torch.uint8)
                request = self.comm.Irecv(
                    [buf.numpy(), self.MPI.BYTE], source=0, tag=P.TAG_MODEL
                )
                self._add_version(version, message["schema"], buf, request=request)
                self.stats["bytes_in"] += int(message["nbytes"])
            with self.cv:
                if version not in self.versions:
                    raise RuntimeError(
                        f"Task {message['tid']} needs global model version "
                        f"{version}, which this worker does not hold."
                    )
                self.refs[version] = self.refs.get(version, 0) + 1
                heapq.heappush(
                    self.heap, (message["deadline"], message["tid"], message)
                )
                self.cv.notify()
        elif kind == P.BCAST:
            buf = torch.empty(message["nbytes"], dtype=torch.uint8)
            request = self.comm.Ibcast([buf.numpy(), self.MPI.BYTE], root=0)
            self._add_version(
                message["version"], message["schema"], buf, request=request
            )
            self.stats["bytes_in"] += int(message["nbytes"])
        elif kind == P.FETCH:
            buf = self.results.pop(message["tid"])
            request = self.comm.Isend(
                [buf.numpy(), self.MPI.BYTE], dest=0, tag=P.TAG_RESULT
            )
            self.sends.append((request, buf))
            self.stats["bytes_out"] += int(buf.numel())
        elif kind == P.DROP:
            self._drop(message["tid"])
        else:
            raise RuntimeError(f"Unexpected {kind!r} message from the coordinator")

    def _add_version(self, version, schema, buf, ready=False, request=None) -> None:
        with self.cv:
            self.versions[version] = {
                "schema": schema,
                "buf": buf,
                "ready": ready or request is None,
            }
            self.refs.setdefault(version, 0)
            if version > self.newest:
                self.newest = version
                self._evict_locked()
        if request is not None:
            self.version_requests.append((version, request, buf))

    def _evict_locked(self) -> None:
        """Free versions that are older than the newest and used by no queued task."""
        for version in list(self.versions):
            if version != self.newest and self.refs.get(version, 0) == 0:
                del self.versions[version]
                self.refs.pop(version, None)

    def _drop(self, tid: int) -> None:
        with self.cv:
            for index, (_, queued_tid, task) in enumerate(self.heap):
                if queued_tid == tid:
                    self.heap.pop(index)
                    heapq.heapify(self.heap)
                    self.refs[task["version"]] -= 1
                    self._evict_locked()
                    self.stats["dropped"] += 1
                    return
            if self.running == tid:
                self.dropped_running.add(tid)
                self.stats["dropped"] += 1
                return
        if self.results.pop(tid, None) is not None:
            self.stats["dropped"] += 1

    def _drain_outbox(self) -> bool:
        drained = False
        while True:
            try:
                kind, task, payload = self.outbox.get_nowait()
            except queue.Empty:
                return drained
            drained = True
            tid = task["tid"]
            if kind == "error":
                self._send_ctrl(
                    {
                        "kind": P.ERROR,
                        "tid": tid,
                        "rank": self.rank,
                        "traceback": payload,
                    }
                )
                continue
            # Count every task trained, including one dropped while it ran.
            self.stats["tasks"] += 1
            self.stats["busy_seconds"] += payload["busy"]
            if tid in self.dropped_running:
                self.dropped_running.discard(tid)
                continue
            self.results[tid] = payload["buf"]
            self._send_ctrl(
                {
                    "kind": P.DONE,
                    "tid": tid,
                    "meta": payload["meta"],
                    "schema": payload["schema"],
                    "nbytes": int(payload["buf"].numel()),
                    "busy": payload["busy"],
                }
            )

    def _send_ctrl(self, message: Dict) -> None:
        request = self.comm.isend(message, dest=0, tag=P.TAG_CTRL)
        self.sends.append((request, None))

    def _test_requests(self) -> bool:
        completed = False
        if self.version_requests:
            remaining = []
            for version, request, buf in self.version_requests:
                if request.Test():
                    completed = True
                    with self.cv:
                        entry = self.versions.get(version)
                        if entry is not None:
                            entry["ready"] = True
                            self.cv.notify_all()
                else:
                    remaining.append((version, request, buf))
            self.version_requests = remaining
        if self.sends:
            remaining = []
            for request, keep in self.sends:
                if request.Test():
                    completed = True
                else:
                    remaining.append((request, keep))
            self.sends = remaining
        return completed

    def _shutdown(self) -> None:
        with self.cv:
            self.stopping = True
            self.stats["dropped"] += len(self.heap)
            self.heap.clear()
            self.cv.notify_all()
        self.thread.join()
        # Every model payload announced before SHUTDOWN has a posted receive, and
        # every FETCHed result has a posted send: finish them before saying BYE.
        pending = [request for _, request, _ in self.version_requests]
        pending += [request for request, _ in self.sends]
        self.MPI.Request.Waitall(pending)
        self.version_requests, self.sends = [], []
        self.comm.send(
            {"kind": P.BYE, "rank": self.rank, "stats": self.stats},
            dest=0,
            tag=P.TAG_CTRL,
        )
        self.node_comm.Free()
        self.worker_comm.Free()
        self.comm.Free()

    # ---------- training thread ----------
    def _train_loop(self) -> None:
        if self.device.startswith("cuda"):
            import torch

            torch.cuda.set_device(self.device)  # the current device is per thread
        while True:
            with self.cv:
                task = None
                while task is None:
                    if self.stopping:
                        return
                    task = self._pop_runnable_locked()
                    if task is None:
                        self.cv.wait(timeout=0.5)
                version = task["version"]
                entry = self.versions[version]
                self.refs[version] -= 1
                self._evict_locked()
                self.running = task["tid"]
            try:
                payload = self._train(task, entry["schema"], entry["buf"])
                item = ("done", task, payload)
            except BaseException:
                item = ("error", task, traceback.format_exc())
            del entry
            with self.cv:
                self.running = None
            self.outbox.put(item)

    def _pop_runnable_locked(self):
        """Pop the earliest-deadline task whose global model has arrived."""
        if not self.heap:
            return None
        if self.versions[self.heap[0][2]["version"]]["ready"]:
            return heapq.heappop(self.heap)[2]
        # Rare: the earliest task's model is still in transit; run the next one.
        for item in sorted(self.heap):
            if self.versions[item[2]["version"]]["ready"]:
                self.heap.remove(item)
                heapq.heapify(self.heap)
                return item[2]
        return None

    def _train(self, task: Dict, schema, buf) -> Dict:
        client = self.clients[task["cid"]]
        state = unpack_state(schema, buf)
        started = time.perf_counter()
        with rng_guard(task["seed"]):
            client.load_parameters(state)
            client.train()
            result = client.get_parameters()
        busy = time.perf_counter() - started
        del state
        if isinstance(result, tuple):
            model, meta = result[0], dict(result[1])
        else:
            model, meta = result, {}
        if self.offload and self.device != "cpu":
            module = getattr(getattr(client, "trainer", None), "model", None)
            if module is not None and hasattr(module, "to"):
                module.to("cpu")
        packed_schema, packed = pack_state(model)
        P.check_payload_size(int(packed.numel()))
        return {"meta": meta, "schema": packed_schema, "buf": packed, "busy": busy}
