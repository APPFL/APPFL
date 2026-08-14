"""Virtual-time synchronous FL simulator driver (v3).

Two modes:
  - count:  dispatch M clients, aggregate first K completions (over-selection)
  - window: dispatch M clients, aggregate all arriving within window_duration

Both modes support max_wait_time (hard deadline) and min_responses (skip round
if fewer arrive).

Inherits BaseSimDriver for common state, utilities, and model init.
"""

import random
from typing import Dict, List, Optional

from .base_sim_driver import BaseSimDriver


class SyncSimDriver(BaseSimDriver):
    """Virtual-time synchronous FL simulation driver.

    Supports two barrier modes: **count** (aggregate first K of M completions)
    and **window** (aggregate all arrivals within a fixed time window).
    """

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
    ):
        super().__init__(
            server_agent,
            client_agents,
            profiles,
            max_concurrency=participants_per_round,
            logger=logger,
            seed=seed,
            base_step_time=base_step_time,
            eval_every=eval_every,
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
        """Select M participants for the current round."""
        all_ids = list(self.clients.keys())
        participant_count = min(self.K, len(all_ids))
        return random.sample(all_ids, participant_count)

    # ---------- duration computation ----------
    def _compute_completions(self, selected, t_start):
        """
        Train all selected clients and compute their virtual completion times.

        :param selected: List of selected client IDs.
        :param t_start: Round start virtual time.
        :return: List of completion dicts, sorted by completion_time.
        """
        self.active = set(selected)
        self._max_active = max(self._max_active, len(self.active))
        completions = []

        global_params = self.server.get_parameters(serial_run=True)
        for cid in selected:
            client = self.clients[cid]
            profile = self.profiles[cid]
            client.load_parameters(global_params)
            client.train()
            result = client.get_parameters()
            if isinstance(result, tuple):
                local_model, meta = result[0], dict(result[1])
            else:
                local_model, meta = result, {}
            steps = int(meta.get("current_local_steps", 0))
            cps = (
                float(self.base_step_time)
                if self.base_step_time is not None
                else float(meta.get("compute_second_per_step", 0.0))
            )
            comp = profile.compute_time(cps, steps)
            comm = profile.comm_time(self._model_bytes)
            duration = comp + comm
            completions.append(
                {
                    "cid": cid,
                    "duration": duration,
                    "completion_time": t_start + duration,
                    "local_model": local_model,
                    "meta": meta,
                }
            )

        completions.sort(key=lambda x: x["completion_time"])
        self.active.clear()
        return completions

    # ---------- barrier logic ----------
    def _apply_barrier(self, completions, t_start):
        """
        Apply the configured barrier to determine accepted/discarded clients.

        :param completions: Sorted list of completion dicts.
        :param t_start: Round start virtual time.
        :return: Tuple of (accepted, discarded, t_barrier).
        """
        if self.mode == "count":
            return self._barrier_count(completions, t_start)
        return self._barrier_window(completions, t_start)

    def _barrier_count(self, completions, t_start):
        """Count barrier: accept first K completions, subject to max_wait_time."""
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
        """Window barrier: accept all completions within window_duration."""
        deadline = t_start + self.window_duration
        accepted = [c for c in completions if c["completion_time"] <= deadline]
        discarded = [c for c in completions if c["completion_time"] > deadline]

        if len(accepted) < self.min_responses:
            if self.max_wait_time:
                hard_deadline = t_start + self.max_wait_time
                accepted = [
                    c for c in completions if c["completion_time"] <= hard_deadline
                ]
                discarded = [
                    c for c in completions if c["completion_time"] > hard_deadline
                ]
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
        """Aggregate local models from accepted clients via the server agent."""
        self.server.scheduler.num_clients = len(accepted)
        for i, c in enumerate(accepted):
            local_model = {
                k: v.cpu() if hasattr(v, "cpu") else v
                for k, v in c["local_model"].items()
            }
            meta = dict(c["meta"])
            meta["virtual_time"] = self.virtual_time
            blocking = i == len(accepted) - 1
            self.server.global_update(
                client_id=c["cid"], local_model=local_model, blocking=blocking, **meta
            )

    # ---------- recording ----------
    def _record_round(self, t_start, t_barrier, accepted, discarded, skipped):
        """Append a round record to ``self.history``."""
        round_comm = self._model_bytes * 2 * len(accepted)
        self._total_comm_bytes += round_comm
        self.history.append(
            {
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
            }
        )

    # ---------- one round ----------
    def _run_round(self):
        """Execute one synchronous FL round: select, train, barrier, aggregate."""
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
                f"{len(completions)} dispatched, 0 met deadline"
            )
            self._record_round(t_start, t_barrier, [], discarded, skipped=True)
            return False

        self._aggregate_round(accepted)

        self._record_round(t_start, t_barrier, accepted, discarded, skipped=False)

        if self.eval_every and (self._round + 1) % self.eval_every == 0:
            global_model = self.server.get_parameters(serial_run=True)
            g = self._global_eval(global_model)
            if g is not None:
                self.history[-1]["global_val_loss"] = g[0]
                self.history[-1]["global_val_accuracy"] = g[1]
                self.logger.info(
                    f"[round={self._round}] GLOBAL "
                    f"val_acc={g[1]:.2f} val_loss={g[0]:.4f}"
                )

        round_comm_mb = self.history[-1]["comm_bytes"] / 1e6
        self.logger.info(
            f"[round={self._round}] vt={t_barrier:9.2f} "
            f"accepted={len(accepted)}/{len(selected)} "
            f"round_dur={t_barrier - t_start:.2f} "
            f"comm={round_comm_mb:.1f}MB"
        )
        return True

    # ---------- main loop ----------
    def run(self):
        """Run the sync simulation for target_rounds, skipping rounds with no available clients."""
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
                        f"(completed {self._round}/{self._target_rounds})"
                    )
                    break
            if self.server.training_finished():
                break
        total_comm_gb = self._total_comm_bytes / 1e9
        self.logger.info(
            f"Simulation finished: rounds={self._round}, "
            f"skipped={self._skipped_rounds}, "
            f"virtual_time={self.virtual_time:.2f}, "
            f"total_comm={total_comm_gb:.2f}GB"
        )

    # ---------- verification (override) ----------
    def verify(self, target_rounds):
        """
        Run sync-specific post-simulation invariant checks.

        :param target_rounds: Expected number of completed rounds.
        :return: Dict of check_name → bool (pass/fail) or int (info).
        """
        non_skipped = [r for r in self.history if not r["skipped"]]
        checks = {
            "monotonic_round_starts": all(
                self.history[i]["t_start"] < self.history[i + 1]["t_start"]
                for i in range(len(self.history) - 1)
            ),
            "completed_rounds==target": len(non_skipped) == target_rounds,
            "barrier_respected": all(
                r["accepted_count"] >= self.min_responses for r in non_skipped
            ),
            "durations_positive": all(r["duration"] > 0 for r in self.history),
            "concurrency<=M": self._max_active <= self.K,
        }
        if self._skipped_rounds > 0:
            checks["skipped_rounds"] = self._skipped_rounds
        return checks
