"""Virtual-time synchronous FL simulator driver (v3).

Two modes:
  - count:  dispatch M clients, aggregate first K completions (over-selection)
  - window: dispatch M clients, aggregate all arriving within window_duration

Both modes support max_wait_time (hard deadline) and min_responses (skip round
if fewer arrive).  All v2 timing models (CommModel, ComputeModel,
AvailabilityModel, timing-only) work unchanged.

Inherits BaseSimDriver for common state, utilities, and model init.
"""

import random
from typing import Dict, List, Optional

from .base_sim_driver import BaseSimDriver


class SyncSimDriver(BaseSimDriver):
    def __init__(
        self,
        server_agent,
        client_agents: List,
        profiles: Dict,
        participants_per_round: int,
        logger,
        seed: int = 42,
        base_step_time: Optional[float] = None,
        eval_every: int = 0,
        # sync
        mode: str = "count",
        min_responses: Optional[int] = None,
        max_wait_time: Optional[float] = None,
        window_duration: Optional[float] = None,
        target_rounds: int = 100,
        # v2
        availability_model=None,
        timeout_model=None,
        shared_pool=None,
        timing_only: bool = False,
        num_local_steps: int = 20,
        calibration_epochs: Optional[int] = None,
    ):
        super().__init__(
            server_agent, client_agents, profiles,
            max_concurrency=participants_per_round,
            logger=logger, seed=seed, base_step_time=base_step_time,
            eval_every=eval_every,
            availability_model=availability_model,
            timeout_model=timeout_model, shared_pool=shared_pool,
            timing_only=timing_only, num_local_steps=num_local_steps,
            target_epochs=target_rounds,
            calibration_epochs=calibration_epochs,
        )
        self.mode = mode
        self.min_responses = min_responses if min_responses is not None else self.K
        self.max_wait_time = max_wait_time
        self.window_duration = window_duration
        self._target_rounds = target_rounds
        self._round = 0
        self._skipped_rounds = 0

        if mode == "window" and window_duration is None:
            raise ValueError("mode='window' requires window_duration")

        self.logger.info(
            f"  sync mode={mode}, M={self.K}, min_responses={self.min_responses}, "
            f"max_wait={max_wait_time}, window={window_duration}, "
            f"target_rounds={target_rounds}"
        )

    # ---------- participant selection ----------
    def _select_participants(self):
        all_ids = list(self.clients.keys())
        if self.availability_model:
            available = [c for c in all_ids
                         if self.availability_model.available(c, self.virtual_time)]
            self._avail_skips += len(all_ids) - len(available)
        else:
            available = all_ids
        M = min(self.K, len(available))
        return random.sample(available, M) if M > 0 else []

    # ---------- duration computation ----------
    def _compute_completions(self, selected, t_start):
        self.active = set(selected)
        self._max_active = max(self._max_active, len(self.active))
        completions = []

        if self.timing_only:
            cps = float(self.base_step_time)
            steps = self._num_local_steps
            for cid in selected:
                profile = self.profiles[cid]
                comp = profile.compute_time(cps, steps)
                comm = profile.comm_time(self._model_bytes, **self._comm_kwargs())
                dur = comp + comm
                completions.append({
                    "cid": cid, "duration": dur,
                    "completion_time": t_start + dur,
                    "local_model": None, "meta": {},
                })
        else:
            global_params = self.server.get_parameters(serial_run=True)
            for cid in selected:
                client = self.clients[cid]
                profile = self.profiles[cid]
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
                comm = profile.comm_time(self._model_bytes, **self._comm_kwargs())
                dur = comp + comm
                completions.append({
                    "cid": cid, "duration": dur,
                    "completion_time": t_start + dur,
                    "local_model": local_model, "meta": meta,
                })

        completions.sort(key=lambda x: x["completion_time"])
        self.active.clear()
        return completions

    # ---------- barrier logic ----------
    def _apply_barrier(self, completions, t_start):
        if self.mode == "count":
            return self._barrier_count(completions, t_start)
        return self._barrier_window(completions, t_start)

    def _barrier_count(self, completions, t_start):
        K = self.min_responses
        if len(completions) < K:
            t = completions[-1]["completion_time"] if completions else t_start
            return completions, [], t

        accepted = completions[:K]
        discarded = completions[K:]
        t_barrier = accepted[-1]["completion_time"]

        if self.max_wait_time and t_barrier > t_start + self.max_wait_time:
            deadline = t_start + self.max_wait_time
            accepted = [c for c in completions if c["completion_time"] <= deadline]
            discarded = [c for c in completions if c["completion_time"] > deadline]
            if len(accepted) < self.min_responses:
                return [], completions, deadline
            t_barrier = deadline

        return accepted, discarded, t_barrier

    def _barrier_window(self, completions, t_start):
        deadline = t_start + self.window_duration
        accepted = [c for c in completions if c["completion_time"] <= deadline]
        discarded = [c for c in completions if c["completion_time"] > deadline]

        if len(accepted) < self.min_responses:
            if self.max_wait_time:
                hard_deadline = t_start + self.max_wait_time
                accepted = [c for c in completions if c["completion_time"] <= hard_deadline]
                discarded = [c for c in completions if c["completion_time"] > hard_deadline]
                if len(accepted) < self.min_responses:
                    return [], completions, hard_deadline
                t_barrier = accepted[-1]["completion_time"]
            else:
                return [], completions, deadline
        else:
            t_barrier = deadline

        return accepted, discarded, t_barrier

    # ---------- aggregation ----------
    def _aggregate_round(self, accepted):
        self.server.scheduler.num_clients = len(accepted)
        for i, c in enumerate(accepted):
            local_model = {k: v.cpu() if hasattr(v, 'cpu') else v
                           for k, v in c["local_model"].items()}
            meta = dict(c["meta"])
            meta["virtual_time"] = self.virtual_time
            blocking = (i == len(accepted) - 1)
            self.server.global_update(
                client_id=c["cid"], local_model=local_model,
                blocking=blocking, **meta)

    # ---------- recording ----------
    def _record_round(self, t_start, t_barrier, accepted, discarded, skipped):
        round_comm = self._model_bytes * 2 * len(accepted)
        self._total_comm_bytes += round_comm
        self.history.append({
            "round": self._round,
            "vtime": t_barrier,
            "t_start": t_start,
            "t_barrier": t_barrier,
            "accepted_count": len(accepted),
            "discarded_count": len(discarded),
            "accepted_cids": [c["cid"] for c in accepted],
            "duration": t_barrier - t_start,
            "skipped": skipped,
            "staleness": 0,
            "completion_ok": True,
            "comm_bytes": round_comm,
        })

    # ---------- one round ----------
    def _run_round(self):
        t_start = self.virtual_time
        selected = self._select_participants()

        if not selected:
            step = self.max_wait_time or self.window_duration or 10.0
            self.virtual_time = t_start + step
            self.logger.info(f"[round={self._round}] SKIP — no available clients")
            self._record_round(t_start, self.virtual_time, [], [], skipped=True)
            return False

        completions = self._compute_completions(selected, t_start)
        accepted, discarded, t_barrier = self._apply_barrier(completions, t_start)

        self.virtual_time = t_barrier

        if not accepted:
            self.logger.info(
                f"[round={self._round}] SKIP — "
                f"{len(completions)} dispatched, 0 met deadline")
            self._record_round(t_start, t_barrier, [], discarded, skipped=True)
            return False

        if not self.timing_only:
            self._aggregate_round(accepted)

        self._record_round(t_start, t_barrier, accepted, discarded, skipped=False)

        if self.eval_every and (self._round + 1) % self.eval_every == 0:
            if not self.timing_only:
                global_model = self.server.get_parameters(serial_run=True)
                g = self._global_eval(global_model)
                if g is not None:
                    self.history[-1]["global_val_loss"] = g[0]
                    self.history[-1]["global_val_accuracy"] = g[1]
                    self.logger.info(
                        f"[round={self._round}] GLOBAL "
                        f"val_acc={g[1]:.2f} val_loss={g[0]:.4f}")

        round_comm_mb = self.history[-1]["comm_bytes"] / 1e6
        self.logger.info(
            f"[round={self._round}] vt={t_barrier:9.2f} "
            f"accepted={len(accepted)}/{len(selected)} "
            f"round_dur={t_barrier - t_start:.2f} "
            f"comm={round_comm_mb:.1f}MB")
        return True

    # ---------- main loop ----------
    def run(self):
        self._calibrate()
        max_skips = self._target_rounds * 3
        while self._round < self._target_rounds:
            success = self._run_round()
            if success:
                self._round += 1
            else:
                self._skipped_rounds += 1
                if self._skipped_rounds >= max_skips:
                    self.logger.warning(
                        f"Stopping: {self._skipped_rounds} consecutive skips "
                        f"(completed {self._round}/{self._target_rounds})")
                    break
            if not self.timing_only and self.server.training_finished():
                break
        total_comm_gb = self._total_comm_bytes / 1e9
        self.logger.info(
            f"Simulation finished: rounds={self._round}, "
            f"skipped={self._skipped_rounds}, "
            f"virtual_time={self.virtual_time:.2f}, "
            f"total_comm={total_comm_gb:.2f}GB")

    # ---------- verification (override) ----------
    def verify(self, target_rounds):
        non_skipped = [r for r in self.history if not r["skipped"]]
        checks = {
            "monotonic_round_starts": all(
                self.history[i]["t_start"] < self.history[i + 1]["t_start"]
                for i in range(len(self.history) - 1)
            ),
            "completed_rounds==target": len(non_skipped) == target_rounds,
            "barrier_respected": all(
                r["accepted_count"] >= self.min_responses
                for r in non_skipped
            ),
            "durations_positive": all(r["duration"] > 0 for r in self.history),
            "concurrency<=M": self._max_active <= self.K,
        }
        if self._skipped_rounds > 0:
            checks["skipped_rounds"] = self._skipped_rounds
        if self._avail_skips > 0:
            checks["availability_skips"] = self._avail_skips
        return checks
