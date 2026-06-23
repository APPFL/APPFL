"""Virtual-time asynchronous FL simulator driver (v1, AFL-Lib-equivalent).

Runs N clients on a single CPU/GPU *sequentially*, but reproduces asynchronous
client *arrival order* via a virtual-time event queue (min-heap). Logical time
jumps to the next completion event; it is decoupled from wall-clock.

This is the async counterpart of `examples/serial/run_serial.py` (which is
synchronous-only). It does NOT subclass any APPFL communicator; it is a plain
driver that reuses `ServerAgent` / `ClientAgent` / scheduler / aggregator.

Design facts verified against APPFL source (2026-06-12):
  - `ServerAgent.global_update(..., blocking=False)` with AsyncScheduler returns a
    dict immediately (Future only for buffered schedulers -> resolved instantly here).
  - Trainer metadata carries `compute_second_per_step`, `current_local_steps`,
    `val_accuracy` (with do_validation=True).
  - Downlink (load latest global) happens at train_start, not at completion.
"""

import heapq
import random
from typing import Dict, List, Optional


class AsyncSimDriver:
    def __init__(
        self,
        server_agent,
        client_agents: List,
        profiles: Dict,
        max_concurrency: int,
        logger,
        seed: int = 42,
        base_step_time: Optional[float] = None,
        eval_every: int = 0,
    ):
        random.seed(seed)
        self.server = server_agent
        self.clients = {c.get_id(): c for c in client_agents}
        self.profiles = profiles
        self.K = max(1, int(max_concurrency))
        self.logger = logger
        # If set, use a fixed per-step compute time (-> fully deterministic virtual time).
        # If None, use the trainer's *measured* compute_second_per_step (realistic, AFL-Lib
        # style, but virtual times vary slightly run-to-run since it is a wall-clock measurement).
        self.base_step_time = base_step_time
        # Evaluate the GLOBAL model on the server validation set every `eval_every`
        # completions (0 = disabled). Gives a true convergence curve (vs per-client local val).
        self.eval_every = int(eval_every) if eval_every else 0

        self.virtual_time = 0.0
        self.queue: List = []          # min-heap: (vtime, seq, etype, cid)
        self._seq = 0                  # monotonic tiebreak (deterministic, fair)
        self.active = set()            # client ids currently "in flight"
        self._pending: Dict = {}       # cid -> stashed (model, meta, dispatch_time, ...)
        self.history: List[Dict] = []  # per-completion log records
        # ---- verification instrumentation (proves virtual-time bookkeeping is correct) ----
        self._max_active = 0           # peak concurrent in-flight clients
        self._mono_violations = 0      # times virtual_time went backwards
        self._prev_vt = None

        # model size in bytes (for comm time), from initial global model
        init_model = self.server.get_parameters(serial_run=True)
        self._model_bytes = self._state_bytes(init_model)
        for c in client_agents:
            c.load_parameters(init_model)

        self.logger.info(
            f"AsyncSimDriver init: clients={len(self.clients)}, K={self.K}, "
            f"model_bytes={self._model_bytes} (~{self._model_bytes / 1e6:.2f} MB), seed={seed}"
        )

    # ---------- helpers ----------
    @staticmethod
    def _state_bytes(state) -> int:
        total = 0
        if hasattr(state, "values"):
            for v in state.values():
                if hasattr(v, "numel") and hasattr(v, "element_size"):
                    total += v.numel() * v.element_size()
        return total

    def _push(self, vtime: float, etype: str, cid: str):
        heapq.heappush(self.queue, (vtime, self._seq, etype, cid))
        self._seq += 1

    def _cur_epoch(self) -> int:
        return self.server.scheduler.get_num_global_epochs()

    def _dispatch_idle(self):
        """Fill up to K concurrent in-flight clients by sampling idle ones."""
        need = self.K - len(self.active)
        if need <= 0:
            return
        idle = [cid for cid in self.clients if cid not in self.active]
        for cid in random.sample(idle, min(need, len(idle))):
            self.active.add(cid)
            self._push(self.virtual_time, "train_start", cid)
        self._max_active = max(self._max_active, len(self.active))

    # ---------- main loop ----------
    def run(self):
        self._dispatch_idle()
        while not self.server.training_finished() and self.queue:
            vtime, _, etype, cid = heapq.heappop(self.queue)
            self.virtual_time = vtime
            if self._prev_vt is not None and vtime < self._prev_vt - 1e-9:
                self._mono_violations += 1
            self._prev_vt = vtime
            if etype == "train_start":
                self._handle_train_start(cid, vtime)
            elif etype == "train_complete":
                self._handle_train_complete(cid, vtime)
        self.logger.info(
            f"Simulation finished: virtual_time={self.virtual_time:.2f}, "
            f"global_epochs={self._cur_epoch()}, completions={len(self.history)}"
        )

    def _handle_train_start(self, cid: str, dispatch_time: float):
        client = self.clients[cid]
        profile = self.profiles[cid]

        if not profile.available(dispatch_time):  # v1: always True
            self._push(dispatch_time, "train_start", cid)
            return

        # Downlink: load the latest global model right before training.
        global_params = self.server.get_parameters(serial_run=True)
        client.load_parameters(global_params)
        epoch_dispatch = self._cur_epoch()

        # Real local training (black-box). Time is modeled separately.
        client.train()
        res = client.get_parameters()
        if isinstance(res, tuple):
            local_model, meta = res[0], dict(res[1])
        else:
            local_model, meta = res, {}

        steps = int(meta.get("current_local_steps", 0))
        # fixed base_step_time -> deterministic; else measured (realistic, non-deterministic)
        if self.base_step_time is not None:
            cps = float(self.base_step_time)
        else:
            cps = float(meta.get("compute_second_per_step", 0.0))
        dur = profile.duration(cps, steps, self._model_bytes)
        completion = dispatch_time + dur

        self._pending[cid] = {
            "local_model": local_model,
            "meta": meta,
            "dispatch_time": dispatch_time,
            "epoch_dispatch": epoch_dispatch,
            "duration": dur,
        }
        self._push(completion, "train_complete", cid)
        self.logger.info(
            f"[vt={dispatch_time:9.2f}] START  {cid:>10} "
            f"dur={dur:8.2f} (compute={profile.compute_time(cps, steps):.2f}"
            f"+comm={profile.comm_time(self._model_bytes):.2f})"
        )

    def _handle_train_complete(self, cid: str, completion_time: float):
        p = self._pending.pop(cid)
        meta = dict(p["meta"])
        val_acc = meta.get("val_accuracy")

        # Pass-through timing info (unused by v1 aggregator; for v2 time-based staleness).
        meta["virtual_time"] = completion_time
        meta["dispatch_time"] = p["dispatch_time"]

        local_model = {k: v.cpu() if hasattr(v, 'cpu') else v for k, v in p["local_model"].items()}

        global_model = self.server.global_update(
            client_id=cid, local_model=local_model, blocking=False, **meta
        )
        if hasattr(global_model, "result"):  # buffered scheduler -> resolves instantly (serial)
            global_model = global_model.result()

        epoch_now = self._cur_epoch()
        # driver-side round-staleness proxy: #global updates during this client's flight
        staleness = epoch_now - p["epoch_dispatch"]
        self.active.discard(cid)

        rec = {
            "vtime": completion_time,
            "cid": cid,
            "epoch": epoch_now,
            "staleness": staleness,
            "val_accuracy": val_acc,
            "duration": p["duration"],
            "dispatch_time": p["dispatch_time"],
            # virtual-time bookkeeping check: completion must equal dispatch + duration
            "completion_ok": abs(p["dispatch_time"] + p["duration"] - completion_time) < 1e-9,
        }
        acc_str = f"{val_acc:.2f}" if isinstance(val_acc, (int, float)) else str(val_acc)
        self.logger.info(
            f"[vt={completion_time:9.2f}] DONE   {cid:>10} epoch={epoch_now:3d} "
            f"staleness={staleness:2d} val_acc={acc_str}"
        )

        # GLOBAL model evaluation on a common server test set (true convergence curve)
        if self.eval_every and (epoch_now % self.eval_every == 0):
            g = self._global_eval(global_model)
            if g is not None:
                rec["global_val_loss"], rec["global_val_accuracy"] = g
                self.logger.info(
                    f"[vt={completion_time:9.2f}] GLOBAL epoch={epoch_now:3d} "
                    f"global_val_acc={g[1]:.2f} global_val_loss={g[0]:.4f}"
                )

        self.history.append(rec)
        self._dispatch_idle()

    def _global_eval(self, global_model):
        """Sync current global params into the server model and validate on the server test set.
        Returns (loss, accuracy) or None if the server has no validation dataset."""
        try:
            self.server.model.load_state_dict(global_model, strict=False)
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"global eval: load_state_dict failed: {e}")
            return None
        return self.server.server_validate()

    def verify(self, target_epochs):
        """Post-run invariant checks proving the virtual-time async bookkeeping is
        correct (not merely that it ran). Returns {check_name: bool}."""
        vts = [r["vtime"] for r in self.history]
        return {
            # virtual time never goes backwards (event queue ordering correct)
            "monotonic_virtual_time": self._mono_violations == 0 and vts == sorted(vts),
            # every completion time == its dispatch time + computed duration
            "completion==dispatch+duration": all(r.get("completion_ok", False) for r in self.history),
            # exactly the requested number of global updates happened
            "completions==target": len(self.history) == target_epochs,
            # concurrency never exceeded K (AFL-Lib MAX_CONCURRENCY semantics)
            "concurrency<=K": self._max_active <= self.K,
            # staleness is a non-negative integer everywhere
            "staleness_nonneg_int": all(
                isinstance(r["staleness"], int) and r["staleness"] >= 0 for r in self.history
            ),
            # every completion took strictly positive virtual time (compute + comm)
            "durations_positive": all(r["duration"] > 0 for r in self.history),
        }
        # NOTE: clients may remain in-flight when the sim stops at the epoch target
        # (normal async early-termination), so a "no dangling active" check is intentionally
        # NOT an invariant here.
