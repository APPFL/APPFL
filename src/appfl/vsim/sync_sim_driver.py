"""Virtual-time synchronous FL simulator driver (v3).

Two modes:
  - count:  dispatch `participants_per_round` clients, aggregate the first
            `min_responses` completions (over-selection)
  - window: dispatch `participants_per_round` clients, aggregate all arriving
            within window_duration

Both modes support max_wait_time (hard deadline) and min_responses (skip round
if fewer arrive).

Inherits BaseSimDriver for common state, utilities, and model init.
"""

from typing import Dict, List, Optional

from .base_sim_driver import BaseSimDriver


class SyncSimDriver(BaseSimDriver):
    """Virtual-time synchronous FL simulation driver.

    Supports two barrier modes: **count** (aggregate the first `min_responses`
    of `participants_per_round` completions) and **window** (aggregate all
    arrivals within a fixed time window).
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
            logger=logger,
            seed=seed,
            base_step_time=base_step_time,
            eval_every=eval_every,
        )
        self.participants_per_round = max(1, int(participants_per_round))
        self.mode = mode
        self.min_responses = (
            min_responses if min_responses is not None else self.participants_per_round
        )
        self.max_wait_time = max_wait_time
        self.window_duration = window_duration
        self._target_rounds = target_rounds
        self._round = 0
        self._skipped_rounds = 0

        if mode == "window" and window_duration is None:
            raise ValueError("mode='window' requires window_duration")

        # A quorum larger than the cohort can never be met, so every round would
        # skip. `_select_participants` caps the cohort at the client count.
        cohort = min(self.participants_per_round, len(self.clients))
        if self.min_responses > cohort:
            raise ValueError(
                f"min_responses={self.min_responses} exceeds the {cohort} client(s) "
                f"dispatched per round (participants_per_round="
                f"{self.participants_per_round}, num_clients={len(self.clients)}), "
                f"so no round could ever reach the quorum."
            )

        # A barrier admits a varying subset each round, so the scheduler must accept
        # a per-round response count. Checked here rather than at the first
        # aggregation: on a scheduler without it, assigning `num_clients` would
        # silently create a dead attribute and the round would never fire.
        scheduler = self.server.scheduler
        if not hasattr(scheduler, "set_num_clients"):
            raise TypeError(
                f"{type(self).__name__} needs a scheduler that can wait for a "
                f"per-round number of responses, i.e. one exposing "
                f"`set_num_clients()` such as SyncScheduler. Got "
                f"{type(scheduler).__name__}. Asynchronous schedulers aggregate on "
                f"every arrival and should be driven with AsyncSimDriver instead."
            )

        self.logger.info(
            f"  sync mode={mode}, participants_per_round={self.participants_per_round}, "
            f"min_responses={self.min_responses}, "
            f"max_wait={max_wait_time}, window={window_duration}, "
            f"target_rounds={target_rounds}"
        )

    # ---------- participant selection ----------
    def _select_participants(self):
        """Select this round's cohort of `participants_per_round` clients."""
        all_ids = list(self.clients.keys())
        participant_count = min(self.participants_per_round, len(all_ids))
        return self._rng.sample(all_ids, participant_count)

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
            _, _, duration = self._client_duration(cid, profile, meta)
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
        """
        Count barrier: release as soon as ``min_responses`` clients have arrived.

        The quorum is guaranteed reachable — the constructor rejects a
        ``min_responses`` larger than the cohort — so a short round can only mean
        the quorum arrives too late. If ``max_wait_time`` is set and the quorum
        arrival falls beyond it, the round is skipped: by definition fewer than
        ``min_responses`` had arrived by the deadline, so there is no smaller set
        to fall back on.

        :param completions: Completion dicts, sorted ascending by completion time.
        :param t_start: Round start virtual time.
        :return: Tuple of (accepted, discarded, t_barrier).
        """
        if len(completions) < self.min_responses:
            return [], completions, t_start

        accepted = completions[: self.min_responses]
        t_barrier = accepted[-1]["completion_time"]

        if self.max_wait_time and t_barrier > t_start + self.max_wait_time:
            return [], completions, t_start + self.max_wait_time

        return accepted, completions[self.min_responses :], t_barrier

    def _barrier_window(self, completions, t_start):
        """
        Window barrier: accept every client arriving within ``window_duration``.

        The clock advances to the end of the window even when all clients arrive
        early, because the server cannot know who else is still coming until the
        window closes. Rounds therefore cost a full window whenever the quorum is
        met — that is the defining property of this mode, not a rounding artifact.

        If fewer than ``min_responses`` arrive, the round extends past the window
        and the barrier releases the moment the quorum is reached, which is the
        earliest point the server may proceed. Extending therefore degrades to the
        count barrier: exactly ``min_responses`` clients are accepted, and the
        clock stops at that arrival rather than running on to ``max_wait_time``.
        Without a ``max_wait_time`` there is nothing to extend into, so the round
        is skipped at the window's end.

        :param completions: Completion dicts, sorted ascending by completion time.
        :param t_start: Round start virtual time.
        :return: Tuple of (accepted, discarded, t_barrier).
        """
        deadline = t_start + self.window_duration
        accepted = [c for c in completions if c["completion_time"] <= deadline]
        if len(accepted) >= self.min_responses:
            return accepted, completions[len(accepted) :], deadline

        if not self.max_wait_time:
            return [], completions, deadline

        # Too few arrived in the window: keep waiting, but only until the quorum
        # is met. `completions` is sorted, so the first `min_responses` entries are
        # the earliest arrivals and the last of them is when the barrier releases.
        hard_deadline = t_start + self.max_wait_time
        reachable = [c for c in completions if c["completion_time"] <= hard_deadline]
        if len(reachable) < self.min_responses:
            return [], completions, hard_deadline

        accepted = completions[: self.min_responses]
        discarded = completions[self.min_responses :]
        return accepted, discarded, accepted[-1]["completion_time"]

    # ---------- aggregation ----------
    def _aggregate_round(self, accepted):
        """
        Aggregate local models from accepted clients via the server agent.

        Only the clients that passed the barrier are submitted, so the scheduler is
        told to fire at that count rather than at the configured federation size.
        The configured value is restored afterwards, leaving the scheduler as found.
        """
        scheduler = self.server.scheduler
        previous_num_clients = scheduler.num_clients
        scheduler.set_num_clients(len(accepted))
        try:
            for i, c in enumerate(accepted):
                local_model = {
                    k: v.cpu() if hasattr(v, "cpu") else v
                    for k, v in c["local_model"].items()
                }
                meta = dict(c["meta"])
                meta["virtual_time"] = self.virtual_time
                blocking = i == len(accepted) - 1
                self.server.global_update(
                    client_id=c["cid"],
                    local_model=local_model,
                    blocking=blocking,
                    **meta,
                )
        finally:
            scheduler.set_num_clients(previous_num_clients)

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
            self.logger.log_content(
                {
                    "round": self._round,
                    "vtime": self.virtual_time,
                    "status": "SKIP",
                    "note": "no clients",
                }
            )
            self._record_round(t_start, self.virtual_time, [], [], skipped=True)
            return False

        completions = self._compute_completions(selected, t_start)
        accepted, discarded, t_barrier = self._apply_barrier(completions, t_start)

        self.virtual_time = t_barrier

        if not accepted:
            self.logger.log_content(
                {
                    "round": self._round,
                    "vtime": t_barrier,
                    "status": "SKIP",
                    "dispatched": len(completions),
                    "accepted": 0,
                    "note": "missed deadline",
                }
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
                    f"global eval @ round {self._round}: "
                    f"val_acc={g[1]:.4f}  val_loss={g[0]:.4f}"
                )

        self.logger.log_content(
            {
                "round": self._round,
                "vtime": t_barrier,
                "status": "OK",
                "dispatched": len(selected),
                "accepted": len(accepted),
                "duration": t_barrier - t_start,
                "comm_MB": self.history[-1]["comm_bytes"] / 1e6,
            }
        )
        return True

    # ---------- main loop ----------
    def run(self):
        """Run the sync simulation for target_rounds, skipping rounds with no available clients."""
        self.logger.log_title(
            [
                "round",
                "vtime",
                "status",
                "dispatched",
                "accepted",
                "duration",
                "comm_MB",
                "note",
            ],
            repeat=True,
        )
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
        self.logger.log_banner(
            "Simulation finished",
            {
                "rounds": self._round,
                "skipped": self._skipped_rounds,
                "virtual_time": f"{self.virtual_time:.2f}s",
                "total_comm": f"{self._total_comm_bytes / 1e9:.2f} GB",
            },
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
            "participants<=participants_per_round": self._max_active
            <= self.participants_per_round,
        }
        if self._skipped_rounds > 0:
            checks["skipped_rounds"] = self._skipped_rounds
        return checks
