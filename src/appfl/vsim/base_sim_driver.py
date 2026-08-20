"""Base class for virtual-time FL simulation drivers.

Extracts common state, utilities, and verification logic shared by the async
and sync drivers. Subclasses implement run() and the
event-handling methods specific to their scheduling model.
"""

import heapq
import random
from typing import Dict, List, Optional

from .logger import ensure_table_logger


class BaseSimDriver:
    """Base class for virtual-time FL simulation drivers.

    Provides common state, heap utilities, global evaluation, and
    post-simulation verification. Subclasses implement ``run()`` and
    event-handling logic for their scheduling model (async or sync).
    """

    def __init__(
        self,
        server_agent,
        client_agents: List,
        profiles: Dict,
        logger,
        seed: int = 42,
        base_step_time: Optional[float] = None,
        eval_every: int = 0,
    ):
        """
        Initialize the simulation driver with server/client agents and models.

        Note that the base class deliberately owns no client-count limit: the
        async driver bounds clients *in flight*, while the sync driver sizes a
        *per-round cohort*. Those are different quantities, so each subclass
        names and verifies its own.

        :param server_agent: APPFL ServerAgent instance.
        :param client_agents: List of APPFL ClientAgent instances.
        :param profiles: Dict mapping client_id to ClientProfile.
        :param logger: Python logger for simulation output.
        :param seed: RNG seed for reproducibility.
        :param base_step_time: Fixed per-step compute time (s); None = measured.
        :param eval_every: Global evaluation frequency (0 = off).
        """
        random.seed(seed)
        self.server = server_agent
        self.clients = {c.get_id(): c for c in client_agents}
        self.profiles = profiles
        self.logger = ensure_table_logger(logger)
        self.base_step_time = base_step_time
        self.eval_every = int(eval_every) if eval_every else 0

        self.virtual_time = 0.0
        self.queue: List = []
        self._seq = 0
        self.active = set()
        self._pending: Dict = {}
        self.history: List[Dict] = []

        self._max_active = 0
        self._mono_violations = 0
        self._prev_vt = None
        self._total_comm_bytes = 0

        init_model = self.server.get_parameters(serial_run=True)
        self._model_bytes = self._state_bytes(init_model)
        for client in client_agents:
            client.load_parameters(init_model)

        self.logger.log_banner(
            f"APPFL virtual-time simulation — {type(self).__name__}",
            {
                "clients": len(self.clients),
                "model": f"{self._model_bytes / 1e6:.2f} MB",
                "step_time": "measured" if base_step_time is None else f"{base_step_time}s",
                "eval_every": self.eval_every or "off",
                "seed": seed,
            },
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

    # ---------- verification ----------
    def verify(self, target_epochs):
        """
        Run post-simulation invariant checks common to every driver.

        Subclasses add whatever bound applies to their own client-count limit.

        :param target_epochs: Expected number of completed global updates.
        :return: Dict of check_name → bool (pass/fail) or int (info).
        """
        vts = [r["vtime"] for r in self.history]
        checks = {
            "monotonic_virtual_time": self._mono_violations == 0 and vts == sorted(vts),
            "completion==dispatch+duration": all(
                r.get("completion_ok", False) for r in self.history
            ),
            "completions==target": len(self.history) == target_epochs,
            "staleness_nonneg_int": all(
                isinstance(r["staleness"], int) and r["staleness"] >= 0
                for r in self.history
            ),
            "durations_positive": all(r["duration"] > 0 for r in self.history),
        }
        return checks

    # ---------- interface ----------
    def run(self):
        """Run the simulation. Subclasses must override this method."""
        raise NotImplementedError
