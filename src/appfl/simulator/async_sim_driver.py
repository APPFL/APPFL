"""Virtual-time asynchronous FL simulator driver.

v1: Runs N clients on a single CPU/GPU *sequentially*, but reproduces asynchronous
client *arrival order* via a virtual-time event queue (min-heap).

v2 additions:
  - AvailabilityModel: dispatch-level dropout (permanent, session, correlated)
  - TimeoutModel: discard slow completions before aggregation
  - CommModel integration: jitter, shared BW pool, asymmetric BW, TCP overhead
  - ComputeModel integration: device profiles, FLOPs-based compute time
  - Timing-only mode: skip actual training, compute virtual time from profiles only
  All are injected via constructor params; all are optional for v1 backward compat.

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

from .base_sim_driver import BaseSimDriver


class AsyncSimDriver(BaseSimDriver):
    """Virtual-time asynchronous FL simulation driver.

    Uses a min-heap event queue to reproduce asynchronous client arrival order
    on a single device.  Supports FedAsync, FedBuff, and other APPFL async
    aggregators without modification.
    """

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
        # v2
        availability_model=None,
        timeout_model=None,
        shared_pool=None,
        timing_only: bool = False,
        num_local_steps: int = 20,
        target_epochs: Optional[int] = None,
        calibration_epochs: Optional[int] = None,
    ):
        super().__init__(
            server_agent, client_agents, profiles, max_concurrency, logger,
            seed=seed, base_step_time=base_step_time, eval_every=eval_every,
            availability_model=availability_model, timeout_model=timeout_model,
            shared_pool=shared_pool, timing_only=timing_only,
            num_local_steps=num_local_steps, target_epochs=target_epochs,
            calibration_epochs=calibration_epochs,
        )

    # ---------- async-specific helpers ----------
    def _cur_epoch(self) -> int:
        """Return the current global epoch count."""
        if self.timing_only:
            return self._timing_epoch
        return self.server.scheduler.get_num_global_epochs()

    def _training_finished(self) -> bool:
        """Check whether the target number of global updates has been reached."""
        if self.timing_only:
            return self._target_epochs is not None and self._timing_epoch >= self._target_epochs
        return self.server.training_finished()

    def _dispatch_idle(self):
        """Fill empty concurrency slots by dispatching idle clients."""
        need = self.K - len(self.active)
        if need <= 0:
            return
        idle = [cid for cid in self.clients if cid not in self.active]

        if self.availability_model is None:
            for cid in random.sample(idle, min(need, len(idle))):
                self.active.add(cid)
                self._push(self.virtual_time, "train_start", cid)
        else:
            random.shuffle(idle)
            dispatched = 0
            for cid in idle:
                if dispatched >= need:
                    break
                if not self.availability_model.available(cid, self.virtual_time):
                    self._avail_skips += 1
                    continue
                self.active.add(cid)
                self._push(self.virtual_time, "train_start", cid)
                dispatched += 1

        self._max_active = max(self._max_active, len(self.active))

    # ---------- main loop ----------
    def run(self):
        """Run the async simulation: pop events from the min-heap until done."""
        self._calibrate()
        self._dispatch_idle()
        while not self._training_finished() and self.queue:
            vtime, _, etype, cid = heapq.heappop(self.queue)
            self.virtual_time = vtime
            if self._prev_vt is not None and vtime < self._prev_vt - 1e-9:
                self._mono_violations += 1
            self._prev_vt = vtime
            if etype == "train_start":
                self._handle_train_start(cid, vtime)
            elif etype == "train_complete":
                self._handle_train_complete(cid, vtime)
        total_comm_gb = self._total_comm_bytes / 1e9
        self.logger.info(
            f"Simulation finished: virtual_time={self.virtual_time:.2f}, "
            f"global_epochs={self._cur_epoch()}, completions={len(self.history)}, "
            f"total_comm={total_comm_gb:.2f}GB"
            + (f", timeout_drops={self._timeout_drops}" if self._timeout_drops else "")
            + (f", avail_skips={self._avail_skips}" if self._avail_skips else "")
        )

    # ---------- event handlers ----------
    def _handle_train_start(self, cid: str, dispatch_time: float):
        """
        Handle a train_start event: download model, train, schedule completion.

        :param cid: Client identifier.
        :param dispatch_time: Virtual time at which the client was dispatched.
        """
        profile = self.profiles[cid]

        if not profile.available(dispatch_time):
            self._push(dispatch_time, "train_start", cid)
            return

        cps = float(self.base_step_time) if self.base_step_time is not None else 0.0
        steps = self._num_local_steps
        epoch_dispatch = self._cur_epoch()

        if self.timing_only:
            comp = profile.compute_time(cps, steps)
            comm_kw = self._comm_kwargs()
            comm = profile.comm_time(self._model_bytes, **comm_kw)
            dur = comp + comm
            completion = dispatch_time + dur

            self._pending[cid] = {
                "local_model": None,
                "meta": {},
                "dispatch_time": dispatch_time,
                "epoch_dispatch": epoch_dispatch,
                "duration": dur,
            }
            self._push(completion, "train_complete", cid)
            self.logger.info(
                f"[vt={dispatch_time:9.2f}] START  {cid:>10} "
                f"dur={dur:8.2f} (compute={comp:.2f}+comm={comm:.2f}) [timing-only]"
            )
            return

        # --- ACTUAL TRAINING ---
        client = self.clients[cid]
        global_params = self.server.get_parameters(serial_run=True)
        client.load_parameters(global_params)

        client.train()
        res = client.get_parameters()
        if isinstance(res, tuple):
            local_model, meta = res[0], dict(res[1])
        else:
            local_model, meta = res, {}

        steps = int(meta.get("current_local_steps", 0))
        if self.base_step_time is not None:
            cps = float(self.base_step_time)
        else:
            cps = float(meta.get("compute_second_per_step", 0.0))

        comp = profile.compute_time(cps, steps)
        comm_kw = self._comm_kwargs()
        comm = profile.comm_time(self._model_bytes, **comm_kw)
        dur = comp + comm
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
            f"dur={dur:8.2f} (compute={comp:.2f}+comm={comm:.2f})"
        )

    def _handle_train_complete(self, cid: str, completion_time: float):
        """
        Handle a train_complete event: aggregate, record, and re-dispatch.

        :param cid: Client identifier.
        :param completion_time: Virtual time at which training completed.
        """
        p = self._pending.pop(cid)

        if self.timeout_model:
            past_durs = [r["duration"] for r in self.history]
            if self.timeout_model.should_discard(p["duration"], past_durs):
                self._timeout_drops += 1
                self.active.discard(cid)
                self.logger.info(
                    f"[vt={completion_time:9.2f}] TIMEOUT {cid:>10} dur={p['duration']:.2f}")
                self._dispatch_idle()
                return

        if self.timing_only:
            self._timing_epoch += 1
            staleness = self._timing_epoch - p["epoch_dispatch"]
            self.active.discard(cid)
            comm_bytes = self._model_bytes * 2
            self._total_comm_bytes += comm_bytes
            rec = {
                "vtime": completion_time,
                "cid": cid,
                "epoch": self._timing_epoch,
                "staleness": staleness,
                "val_accuracy": None,
                "duration": p["duration"],
                "dispatch_time": p["dispatch_time"],
                "completion_ok": abs(p["dispatch_time"] + p["duration"] - completion_time) < 1e-9,
                "comm_bytes": comm_bytes,
            }
            self.logger.info(
                f"[vt={completion_time:9.2f}] DONE   {cid:>10} epoch={self._timing_epoch:3d} "
                f"staleness={staleness:2d} [timing-only]"
            )
            self.history.append(rec)
            self._dispatch_idle()
            return

        # --- ACTUAL AGGREGATION ---
        meta = dict(p["meta"])
        val_acc = meta.get("val_accuracy")
        meta["virtual_time"] = completion_time
        meta["dispatch_time"] = p["dispatch_time"]

        local_model = {k: v.cpu() if hasattr(v, 'cpu') else v for k, v in p["local_model"].items()}

        global_model = self.server.global_update(
            client_id=cid, local_model=local_model, blocking=False, **meta
        )
        if hasattr(global_model, "result"):
            global_model = global_model.result()

        epoch_now = self._cur_epoch()
        staleness = epoch_now - p["epoch_dispatch"]
        self.active.discard(cid)

        comm_bytes = self._model_bytes * 2
        self._total_comm_bytes += comm_bytes
        rec = {
            "vtime": completion_time,
            "cid": cid,
            "epoch": epoch_now,
            "staleness": staleness,
            "val_accuracy": val_acc,
            "duration": p["duration"],
            "dispatch_time": p["dispatch_time"],
            "completion_ok": abs(p["dispatch_time"] + p["duration"] - completion_time) < 1e-9,
            "comm_bytes": comm_bytes,
        }
        acc_str = f"{val_acc:.2f}" if isinstance(val_acc, (int, float)) else str(val_acc)
        self.logger.info(
            f"[vt={completion_time:9.2f}] DONE   {cid:>10} epoch={epoch_now:3d} "
            f"staleness={staleness:2d} val_acc={acc_str}"
        )

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
