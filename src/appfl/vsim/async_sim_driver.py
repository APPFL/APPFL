"""Virtual-time asynchronous FL simulator driver.

v1: Runs N clients on a single CPU/GPU *sequentially*, but reproduces asynchronous
client *arrival order* via a virtual-time event queue (min-heap).

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
        max_in_flight: int,
        logger,
        seed: int = 42,
        base_step_time: Optional[float] = None,
        eval_every: int = 0,
    ):
        """
        :param max_in_flight: Maximum clients dispatched but not yet arrived.
            Nothing runs concurrently — training is strictly serial — so this
            bounds *virtual* outstanding updates, not physical parallelism.
        """
        super().__init__(
            server_agent,
            client_agents,
            profiles,
            logger,
            seed=seed,
            base_step_time=base_step_time,
            eval_every=eval_every,
        )
        self.max_in_flight = max(1, int(max_in_flight))
        self.logger.info(f"  async max_in_flight={self.max_in_flight}")

    # ---------- async-specific helpers ----------
    def _cur_epoch(self) -> int:
        """Return the current global epoch count."""
        return self.server.scheduler.get_num_global_epochs()

    def _training_finished(self) -> bool:
        """Check whether the target number of global updates has been reached."""
        return self.server.training_finished()

    def _dispatch_idle(self):
        """Refill in-flight slots up to `max_in_flight` from the idle pool."""
        need = self.max_in_flight - len(self.active)
        if need <= 0:
            return
        idle = [cid for cid in self.clients if cid not in self.active]

        for cid in random.sample(idle, min(need, len(idle))):
            self.active.add(cid)
            self._push(self.virtual_time, "train_start", cid)

        self._max_active = max(self._max_active, len(self.active))

    # ---------- main loop ----------
    def run(self):
        """Run the async simulation: pop events from the min-heap until done."""
        self.logger.log_title(
            [
                "vtime",
                "event",
                "client",
                "duration",
                "compute",
                "comm",
                "epoch",
                "staleness",
                "val_acc",
            ],
            repeat=True,
        )
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
        self.logger.log_banner(
            "Simulation finished",
            {
                "virtual_time": f"{self.virtual_time:.2f}s",
                "global_epochs": self._cur_epoch(),
                "completions": len(self.history),
                "total_comm": f"{self._total_comm_bytes / 1e9:.2f} GB",
            },
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

        epoch_dispatch = self._cur_epoch()

        # --- ACTUAL TRAINING ---
        client = self.clients[cid]
        # `client_id` lets the aggregator record which global version this client
        # trains from; without it staleness is measured from the client's previous
        # upload, which over-counts whenever a client idles between dispatches.
        global_params = self.server.get_parameters(serial_run=True, client_id=cid)
        client.load_parameters(global_params)

        client.train()
        res = client.get_parameters()
        if isinstance(res, tuple):
            local_model, meta = res[0], dict(res[1])
        else:
            local_model, meta = res, {}

        comp, comm, dur = self._client_duration(cid, profile, meta)
        completion = dispatch_time + dur

        self._pending[cid] = {
            "local_model": local_model,
            "meta": meta,
            "dispatch_time": dispatch_time,
            "epoch_dispatch": epoch_dispatch,
            "duration": dur,
        }
        self._push(completion, "train_complete", cid)
        self.logger.log_content(
            {
                "vtime": dispatch_time,
                "event": "START",
                "client": cid,
                "duration": dur,
                "compute": comp,
                "comm": comm,
            }
        )

    def _handle_train_complete(self, cid: str, completion_time: float):
        """
        Handle a train_complete event: aggregate, record, and re-dispatch.

        :param cid: Client identifier.
        :param completion_time: Virtual time at which training completed.
        """
        p = self._pending.pop(cid)

        # --- ACTUAL AGGREGATION ---
        meta = dict(p["meta"])
        val_acc = meta.get("val_accuracy")
        meta["virtual_time"] = completion_time
        meta["dispatch_time"] = p["dispatch_time"]

        local_model = {
            k: v.cpu() if hasattr(v, "cpu") else v for k, v in p["local_model"].items()
        }

        # Captured before the update: `global_update` increments the epoch counter,
        # so reading it afterwards would make staleness 0 impossible.
        epoch_before = self._cur_epoch()

        global_model = self.server.global_update(
            client_id=cid, local_model=local_model, blocking=False, **meta
        )
        if hasattr(global_model, "result"):
            global_model = global_model.result()

        epoch_now = self._cur_epoch()
        staleness = epoch_before - p["epoch_dispatch"]
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
            "completion_ok": abs(p["dispatch_time"] + p["duration"] - completion_time)
            < 1e-9,
            "comm_bytes": comm_bytes,
        }
        self.logger.log_content(
            {
                "vtime": completion_time,
                "event": "DONE",
                "client": cid,
                "epoch": epoch_now,
                "staleness": staleness,
                "val_acc": val_acc if isinstance(val_acc, (int, float)) else "-",
            }
        )

        if self.eval_every and (epoch_now % self.eval_every == 0):
            g = self._global_eval(global_model)
            if g is not None:
                rec["global_val_loss"], rec["global_val_accuracy"] = g
                self.logger.info(
                    f"global eval @ epoch {epoch_now}: "
                    f"val_acc={g[1]:.4f}  val_loss={g[0]:.4f}"
                )

        self.history.append(rec)
        self._dispatch_idle()

    # ---------- verification ----------
    def verify(self, target_epochs):
        """
        Run the shared invariant checks plus the async in-flight bound.

        :param target_epochs: Expected number of completed global updates.
        :return: Dict of check_name → bool (pass/fail) or int (info).
        """
        checks = super().verify(target_epochs)
        checks["in_flight<=max_in_flight"] = self._max_active <= self.max_in_flight
        return checks
