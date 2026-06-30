"""Base class for virtual-time FL simulation drivers.

Extracts common state, utilities, and verification logic shared by all driver
variants (async, sync, cross-device).  Subclasses implement run() and the
event-handling methods specific to their scheduling model.
"""

import heapq
import random
from typing import Dict, List, Optional


class BaseSimDriver:
    """Base class for virtual-time FL simulation drivers.

    Provides common state, heap utilities, global evaluation, calibration,
    and post-simulation verification.  Subclasses implement ``run()`` and
    event-handling logic for their scheduling model (async or sync).
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
        availability_model=None,
        timeout_model=None,
        shared_pool=None,
        timing_only: bool = False,
        num_local_steps: int = 20,
        target_epochs: Optional[int] = None,
        calibration_epochs: Optional[int] = None,
    ):
        """
        Initialize the simulation driver with server/client agents and models.

        :param server_agent: APPFL ServerAgent instance.
        :param client_agents: List of APPFL ClientAgent instances.
        :param profiles: Dict mapping client_id to ClientProfile.
        :param max_concurrency: K — maximum concurrent in-flight clients.
        :param logger: Python logger for simulation output.
        :param seed: RNG seed for reproducibility.
        :param base_step_time: Fixed per-step compute time (s); None = measured.
        :param eval_every: Global evaluation frequency (0 = off).
        :param availability_model: Dispatch-level dropout model (optional).
        :param timeout_model: Completion-level timeout model (optional).
        :param shared_pool: SharedBandwidthPool for congestion modeling (optional).
        :param timing_only: If True, skip actual training/aggregation.
        :param num_local_steps: Number of local training steps per client.
        :param target_epochs: Target number of global updates.
        :param calibration_epochs: Steps to profile before switching to timing-only.
        """
        random.seed(seed)
        self.server = server_agent
        self.clients = {c.get_id(): c for c in client_agents}
        self.profiles = profiles
        self.K = max(1, int(max_concurrency))
        self.logger = logger
        self.base_step_time = base_step_time
        self.eval_every = int(eval_every) if eval_every else 0

        self.availability_model = availability_model
        self.timeout_model = timeout_model
        self.shared_pool = shared_pool
        self._comm_rng = random.Random(seed + 7) if any(
            getattr(p.comm, "jitter_sigma", 0) > 0 for p in profiles.values()
            if p.comm is not None
        ) else None

        self.timing_only = timing_only
        self._num_local_steps = num_local_steps
        self._target_epochs = target_epochs
        self._timing_epoch = 0
        self._calibration_epochs = calibration_epochs
        if timing_only and base_step_time is None and calibration_epochs is None:
            raise ValueError("timing_only=True requires base_step_time to be set")

        self.virtual_time = 0.0
        self.queue: List = []
        self._seq = 0
        self.active = set()
        self._pending: Dict = {}
        self.history: List[Dict] = []

        self._max_active = 0
        self._mono_violations = 0
        self._prev_vt = None
        self._timeout_drops = 0
        self._avail_skips = 0
        self._total_comm_bytes = 0

        needs_model = not timing_only or calibration_epochs is not None
        if needs_model:
            init_model = self.server.get_parameters(serial_run=True)
            self._model_bytes = self._state_bytes(init_model)
            for c in client_agents:
                c.load_parameters(init_model)
        else:
            self._model_bytes = self._state_bytes(
                self.server.model.state_dict() if hasattr(self.server, 'model') and self.server.model is not None
                else {}
            )

        if calibration_epochs is not None:
            mode_str = f" [CALIBRATION: {calibration_epochs} local steps → timing-only]"
        elif timing_only:
            mode_str = " [TIMING-ONLY]"
        else:
            mode_str = ""
        v2_info = []
        if availability_model is not None:
            v2_info.append(f"availability={type(availability_model).__name__}")
        if timeout_model is not None:
            v2_info.append(f"timeout={timeout_model.timeout or f'q{timeout_model.quantile}'}")
        if shared_pool is not None:
            v2_info.append(f"shared_pool={shared_pool.total_bw}Mbps/{shared_pool.mode}")
        v2_str = f", v2=[{', '.join(v2_info)}]" if v2_info else ""
        self.logger.info(
            f"{type(self).__name__} init{mode_str}: clients={len(self.clients)}, K={self.K}, "
            f"model_bytes={self._model_bytes} (~{self._model_bytes / 1e6:.2f} MB), "
            f"seed={seed}{v2_str}"
        )

    # ---------- utilities ----------
    @staticmethod
    def _state_bytes(state) -> int:
        """
        Compute total bytes of a model state_dict.

        :param state: Model state dict (or empty dict).
        :return: Total byte count across all tensors.
        """
        total = 0
        if hasattr(state, "values"):
            for v in state.values():
                if hasattr(v, "numel") and hasattr(v, "element_size"):
                    total += v.numel() * v.element_size()
        return total

    def _push(self, vtime: float, etype: str, cid: str):
        """
        Push an event onto the min-heap with auto-incrementing tiebreaker.

        :param vtime: Virtual time for the event.
        :param etype: Event type string (e.g. ``"train_start"``).
        :param cid: Client identifier.
        """
        heapq.heappush(self.queue, (vtime, self._seq, etype, cid))
        self._seq += 1

    def _comm_kwargs(self):
        """Build keyword arguments for ``ClientProfile.comm_time()``."""
        kw = {}
        if self.shared_pool is not None:
            kw["num_concurrent"] = len(self.active)
            kw["shared_pool"] = self.shared_pool
        if self._comm_rng is not None:
            kw["rng"] = self._comm_rng
        return kw

    def _global_eval(self, global_model):
        """
        Evaluate the global model on the server's validation set.

        :param global_model: Model state dict to evaluate.
        :return: Tuple of (val_loss, val_accuracy), or None on failure.
        """
        try:
            self.server.model.load_state_dict(global_model, strict=False)
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"global eval: load_state_dict failed: {e}")
            return None
        return self.server.server_validate()

    # ---------- calibration ----------
    def _calibrate(self):
        """Profile all clients for a few local steps, then switch to timing-only mode."""
        if self._calibration_epochs is None:
            return
        cal_steps = min(self._calibration_epochs, self._num_local_steps)
        global_params = self.server.get_parameters(serial_run=True)
        cps_values = []
        self.logger.info(f"Calibration: profiling {len(self.clients)} clients "
                         f"× {cal_steps} local steps...")

        for cid, client in self.clients.items():
            orig_steps = None
            if hasattr(client, 'trainer') and hasattr(client.trainer, 'train_configs'):
                orig_steps = client.trainer.train_configs.num_local_steps
                client.trainer.train_configs.num_local_steps = cal_steps

            client.load_parameters(global_params)
            client.train()
            res = client.get_parameters()

            if orig_steps is not None:
                client.trainer.train_configs.num_local_steps = orig_steps

            if isinstance(res, tuple):
                meta = dict(res[1])
            else:
                meta = {}
            cps = float(meta.get("compute_second_per_step", 0.0))
            if cps > 0:
                cps_values.append(cps)
            self.logger.info(f"  {cid}: cps={cps:.6f}s")

        if not cps_values:
            raise ValueError("Calibration failed: no cps measurements collected")

        mean_cps = sum(cps_values) / len(cps_values)
        std_cps = (sum((x - mean_cps) ** 2 for x in cps_values)
                   / len(cps_values)) ** 0.5
        self.base_step_time = mean_cps
        self.timing_only = True
        self.logger.info(
            f"CALIBRATION DONE: {len(cps_values)} samples, "
            f"mean_cps={mean_cps:.6f}s (std={std_cps:.6f}s) → timing-only"
        )

    # ---------- verification ----------
    def verify(self, target_epochs):
        """
        Run post-simulation invariant checks.

        :param target_epochs: Expected number of completed global updates.
        :return: Dict of check_name → bool (pass/fail) or int (info).
        """
        vts = [r["vtime"] for r in self.history]
        checks = {
            "monotonic_virtual_time": self._mono_violations == 0 and vts == sorted(vts),
            "completion==dispatch+duration": all(r.get("completion_ok", False) for r in self.history),
            "completions==target": len(self.history) == target_epochs,
            "concurrency<=K": self._max_active <= self.K,
            "staleness_nonneg_int": all(
                isinstance(r["staleness"], int) and r["staleness"] >= 0 for r in self.history
            ),
            "durations_positive": all(r["duration"] > 0 for r in self.history),
        }
        if self._timeout_drops > 0:
            checks["timeout_drops"] = self._timeout_drops
        if self._avail_skips > 0:
            checks["availability_skips"] = self._avail_skips
        return checks

    # ---------- interface ----------
    def run(self):
        """Run the simulation. Subclasses must override this method."""
        raise NotImplementedError
