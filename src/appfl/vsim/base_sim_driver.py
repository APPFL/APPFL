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
        compression_ratio: float = 1.0,
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
        :param compression_ratio: Fraction of the raw model size actually sent
            (1.0 = uncompressed, 0.25 = 4x compression). The simulator times the
            bytes on the wire, and cannot know a compressor's real ratio without
            running it, so an enabled compressor must be declared here.
        """
        # A private generator, not `random.seed()`: seeding the module-global RNG
        # here would perturb every other consumer in the process (client data
        # partitioning has already run by this point), and any unrelated draw from
        # it would in turn shift this simulation's dispatch order.
        self._rng = random.Random(seed)
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

        if not 0 < compression_ratio <= 1:
            raise ValueError(
                f"compression_ratio must be in (0, 1]; got {compression_ratio}"
            )
        self.compression_ratio = compression_ratio

        init_model = self.server.get_parameters(serial_run=True)
        self._raw_model_bytes = self._state_bytes(init_model)
        # Transfer times are driven by what actually goes over the wire.
        self._model_bytes = self._raw_model_bytes * compression_ratio
        for client in client_agents:
            client.load_parameters(init_model)

        self.logger.log_banner(
            f"APPFL virtual-time simulation — {type(self).__name__}",
            {
                "clients": len(self.clients),
                "model": f"{self._raw_model_bytes / 1e6:.2f} MB",
                "on_wire": f"{self._model_bytes / 1e6:.2f} MB",
                "step_time": (
                    "measured" if base_step_time is None else f"{base_step_time}s"
                ),
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

    def _client_duration(self, cid: str, profile, meta: Dict):
        """
        Derive a client's virtual duration from its training metadata.

        Raises rather than silently producing a zero-compute round. Only
        ``VanillaTrainer`` reports these keys, so another trainer would otherwise
        yield a run in which every duration is pure communication time. Such a run
        completes normally and passes ``verify()``, because communication keeps
        ``duration > 0``.

        :param cid: Client identifier, used in the error message.
        :param profile: The client's :class:`ClientProfile`.
        :param meta: Metadata returned by the client's trainer.
        :return: Tuple of (compute_time, comm_time, duration), all in seconds.
        """
        steps = meta.get("current_local_steps")
        cps = self.base_step_time
        if cps is None:
            cps = meta.get("compute_second_per_step")
            missing = "compute_second_per_step"
        else:
            missing = None

        if steps is None or float(steps) <= 0:
            self._raise_missing_metadata(cid, "current_local_steps", steps)
        if cps is None or float(cps) <= 0:
            self._raise_missing_metadata(cid, missing or "base_step_time", cps)

        comp = profile.compute_time(float(cps), int(steps))
        comm = profile.comm_time(self._model_bytes)
        return comp, comm, comp + comm

    @staticmethod
    def _raise_missing_metadata(cid: str, key: str, value):
        """Raise a runtime error naming the metadata the duration model needs."""
        raise RuntimeError(
            f"Client {cid} reported {key}={value!r}, so its virtual duration cannot "
            f"be computed. Virtual durations come from trainer metadata: "
            f"`current_local_steps` and `compute_second_per_step`, which "
            f'VanillaTrainer reports in both "step" and "epoch" mode. Use a '
            f"trainer that reports them, or set `base_step_time` under "
            f"`server_configs.simulator` to supply a fixed per-step time (a step "
            f"count is still required from the trainer)."
        )

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
