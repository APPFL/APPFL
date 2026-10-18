"""Base class for virtual-time FL simulation drivers.

Extracts common state, utilities, and verification logic shared by the async
and sync drivers. Subclasses implement run() and the
event-handling methods specific to their scheduling model.

Local training runs in one of two ways. Without an executor, clients train eagerly
in this process and each duration comes from the trainer's own timing metadata.
With a :class:`~appfl.vsim.executor.DeferredExecutor`, training runs elsewhere (for
example on MPI workers), so every duration is computed when the client is
dispatched from a fixed or calibrated per-step time, and the event loop only blocks
when it needs a specific result.
"""

import heapq
import random
from typing import Dict, List, Optional

from .codec import PackedState, unpack_state
from .flat_aggregation import make_flat_aggregation
from .logger import ensure_table_logger
from .seeding import rng_guard, task_seed


class BaseSimDriver:
    """Base class for virtual-time FL simulation drivers.

    Provides common state, heap utilities, global evaluation, and
    post-simulation verification. Subclasses implement ``run()`` and
    event-handling logic for their scheduling model (async or sync).
    """

    #: Kind of flat-buffer aggregation the subclass can use: "async" or "sync".
    _flat_kind: Optional[str] = None

    def __init__(
        self,
        server_agent,
        client_agents: Optional[List],
        profiles: Dict,
        logger,
        seed: int = 42,
        base_step_time: Optional[float] = None,
        eval_every: int = 0,
        compression_ratio: float = 1.0,
        executor=None,
        task_seeding: bool = True,
        flat_aggregation: bool = True,
    ):
        """
        Initialize the simulation driver with server/client agents and models.

        Note that the base class deliberately owns no client-count limit: the
        async driver bounds clients *in flight*, while the sync driver sizes a
        *per-round cohort*. Those are different quantities, so each subclass
        names and verifies its own.

        :param server_agent: APPFL ServerAgent instance.
        :param client_agents: List of APPFL ClientAgent instances. May be ``None``
            when ``executor`` hosts the clients.
        :param profiles: Dict mapping client_id to ClientProfile.
        :param logger: Python logger for simulation output.
        :param seed: RNG seed for reproducibility.
        :param base_step_time: Fixed per-step compute time (s); None = measured
            in-process, or calibrated by ``executor``.
        :param eval_every: Global evaluation frequency (0 = off).
        :param compression_ratio: Fraction of the raw model size actually sent
            (1.0 = uncompressed, 0.25 = 4x compression). The simulator times the
            bytes on the wire, and cannot know a compressor's real ratio without
            running it, so an enabled compressor must be declared here.
        :param executor: Optional :class:`~appfl.vsim.executor.DeferredExecutor`
            that trains clients outside this process. ``None`` trains them here.
        :param task_seeding: Seed the global RNGs per task from (seed, client,
            selection count), so a client's local training does not depend on
            which clients trained before it or where it runs.
        :param flat_aggregation: With FedAsync (async driver) or FedAvg (sync
            driver), keep the global model as one packed buffer and aggregate into
            it in place rather than through the server agent (see
            :mod:`appfl.vsim.flat_aggregation`). Models are bit-identical either
            way; servers it does not cover use the agent regardless.
        """
        # A private generator, not `random.seed()`: seeding the module-global RNG
        # here would perturb every other consumer in the process (client data
        # partitioning has already run by this point), and any unrelated draw from
        # it would in turn shift this simulation's dispatch order.
        self._rng = random.Random(seed)
        self._seed = seed
        self.server = server_agent
        self.executor = executor
        self.task_seeding = bool(task_seeding)
        if executor is None:
            if client_agents is None:
                raise ValueError("client_agents is required when no executor is given")
            self.clients = {c.get_id(): c for c in client_agents}
            self.client_ids = list(self.clients)
        else:
            # Order matters: dispatch sampling draws from this list, so it must be
            # the configured client order, identical on every run.
            self.clients = {}
            self.client_ids = list(executor.client_ids)
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
        self._selections: Dict[str, int] = {}

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
        self._flat = None
        aggregation = "server agent"
        if flat_aggregation and self._flat_kind is not None:
            self._flat, aggregation = make_flat_aggregation(
                self.server, init_model, self._flat_kind, executor
            )
        if executor is None:
            for client in client_agents:
                client.load_parameters(init_model)
            step_source = "measured" if base_step_time is None else f"{base_step_time}s"
        else:
            # Started from the flat global model's buffer, workers pack their results
            # in its layout.
            executor.setup(
                self._flat.params() if self._flat is not None else init_model,
                version=self.server.scheduler.get_num_global_epochs(),
                calibrate=base_step_time is None,
            )
            if base_step_time is None:
                self.base_step_time = self._calibrated_step_time()
                step_source = f"{self.base_step_time:.6g}s (calibrated)"
            else:
                step_source = f"{base_step_time}s"

        self.logger.log_banner(
            f"APPFL virtual-time simulation — {type(self).__name__}",
            {
                "clients": len(self.client_ids),
                "executor": (
                    "in-process" if executor is None else type(executor).__name__
                ),
                "model": f"{self._raw_model_bytes / 1e6:.2f} MB",
                "on_wire": f"{self._model_bytes / 1e6:.2f} MB",
                "step_time": step_source,
                "eval_every": self.eval_every or "off",
                "seed": seed,
                "aggregation": aggregation,
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

    def _calibrated_step_time(self) -> float:
        """Return the executor's calibrated per-step time, or raise if it has none."""
        step_time = self.executor.step_time
        if step_time is None or float(step_time) <= 0:
            raise RuntimeError(
                f"{type(self.executor).__name__} trains clients outside this "
                f"process, so every duration must be known when the client is "
                f"dispatched. Set `base_step_time` under `server_configs.simulator`, "
                f"or use an executor that calibrates one (got step_time="
                f"{step_time!r})."
            )
        return float(step_time)

    def _next_seed(self, cid: str) -> Optional[int]:
        """
        Count one selection of ``cid`` and return the seed for its task.

        Every selection is counted, including clients a sync barrier later
        discards, so a client's k-th task gets the same seed whether or not the
        discarded tasks were physically trained.

        :param cid: Client identifier.
        :return: Task seed, or ``None`` when task seeding is disabled.
        """
        count = self._selections.get(cid, 0)
        self._selections[cid] = count + 1
        return task_seed(self._seed, cid, count) if self.task_seeding else None

    def _train_in_process(self, cid: str, params, seed: Optional[int]):
        """
        Train one client in this process from ``params``.

        :param cid: Client identifier.
        :param params: Global model the client trains from.
        :param seed: Task seed applied around training, or ``None``.
        :return: Tuple of (local_model, meta).
        """
        client = self.clients[cid]
        with rng_guard(seed):
            client.load_parameters(params)
            client.train()
            result = client.get_parameters()
        if isinstance(result, tuple):
            return result[0], dict(result[1])
        return result, {}

    def _apriori_duration(self, cid: str, profile):
        """
        Compute a client's duration at dispatch, before it has trained.

        Uses the executor's census step count with the fixed or calibrated
        per-step time, through the same arithmetic as :meth:`_client_duration`.
        """
        steps = self.executor.expected_steps(cid)
        return self._client_duration(cid, profile, {"current_local_steps": steps})

    def _check_steps(self, cid: str, meta: Dict):
        """
        Fail if a remotely trained client ran a different number of steps.

        The duration was fixed at dispatch from the census, so a mismatch means the
        virtual schedule no longer describes what was trained.
        """
        expected = self.executor.expected_steps(cid)
        reported = meta.get("current_local_steps")
        if reported is None or int(reported) != int(expected):
            raise RuntimeError(
                f"Client {cid} reported current_local_steps={reported!r}, but its "
                f"virtual duration was computed at dispatch from {expected} steps. "
                f"The step count must be predictable before training for "
                f"distributed simulation (for example, not Poisson-sampled DP "
                f"batches)."
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

    # ---------- flat-buffer aggregation ----------
    def _leave_flat_path(self, cid: str):
        """
        Hand aggregation back to the server agent for the rest of the run.

        Called when ``cid``'s local model does not hold exactly the global model's
        tensors, which only the server agent's own rules can aggregate.
        """
        self._flat.sync_server()
        self._flat = None
        self.logger.warning(
            f"{cid}'s local model holds other tensors than the global model; the "
            f"server agent aggregates from here on"
        )

    @staticmethod
    def _as_state_dict(local):
        """Return a local model as a state dict, viewing into it if it is packed."""
        if isinstance(local, PackedState):
            return unpack_state(local.schema, local.buffer)
        return local

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
