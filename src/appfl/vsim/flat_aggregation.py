"""Flat-buffer aggregation on the vsim coordinator.

APPFL's server path, ``ServerAgent.global_update`` and ``get_parameters``, works
tensor by tensor and re-materializes the global model on every update: temporaries
per tensor, a ``load_state_dict`` into the server model, a clone for the return
value and another for the next download, several ``gc.collect`` calls, and, with a
:class:`~appfl.vsim.executor.DeferredExecutor`, a pack of each new version before
sending it. With FedAsyncAggregator + AsyncScheduler or FedAvgAggregator +
SyncScheduler, a driver can instead keep the global model as one packed buffer (see
:mod:`appfl.vsim.codec`) and aggregate into it with a few in-place operations on
flat slices:

- FedAsync writes each update into the local model's own buffer, which then becomes
  the new version: ``L = G + alpha_t * (L - G)`` on the aggregated parameters, while
  every other entry keeps the local value, as FedAsyncAggregator does.
- FedAvg sums each local model's weighted difference into the round's buffer as the
  model arrives, so a sync barrier leaves only the last model and the final
  ``G + sum`` to do.

The operations are the aggregators' own, applied element for element in the same
order (``sub``, ``mul`` and ``add`` rather than a fused ``lerp``), so the global
model is bit-identical to APPFL's path. The global model is laid out with the
aggregated parameters first, then other floating-point tensors, then integers, each
group contiguous. Workers pack every result in the layout of the global model it
trained from, so each group is one slice in every buffer.

A version's buffer is never written once it is published: in-flight sends of it
stay valid without a copy, and the executor reuses the buffer (see
:meth:`~appfl.vsim.executor.DeferredExecutor.recycle`) only after they complete.

The counters that drive a run (the scheduler's epoch count, FedAsync's per-client
download steps) are kept current on every update. The model itself is written back
into the server agent only on demand: :meth:`sync_server` runs before global
evaluation and at the end of a run, and leaves the server as APPFL's own path would.
"""

from typing import Dict, List, Optional, Tuple

from .codec import PackedState, make_layout, pack_state, same_layout, unpack_state

# Groups of entries in the global layout, in layout order. "param" is what the
# aggregators update arithmetically (their own rule: a named parameter that is not
# an int32/int64 tensor); FedAvg averages "float", "int64" and "int32" entries, and
# FedAsync keeps the local value of everything but "param".
_GROUPS = ("param", "float", "int64", "int32", "other")
_DTYPES = {"param": "float32", "float": "float32", "int64": "int64", "int32": "int32"}


def make_flat_aggregation(server, init_model, kind: str, executor=None):
    """
    Return a flat-buffer aggregation standing in for ``server``, if it can.

    :param server: The driver's server agent.
    :param init_model: The initial global model, as ``server.get_parameters``
        returned it.
    :param kind: ``"async"`` (FedAsync) or ``"sync"`` (FedAvg).
    :param executor: The driver's executor, or ``None`` for in-process training.
    :return: Tuple of (:class:`FlatFedAsync`, :class:`FlatFedAvg` or ``None``, and
        a description of the aggregation path for the driver's banner).
    """
    reason = _unsupported_server(server, kind)
    plan = None
    if reason is None:
        plan = _plan_layout(init_model, server.aggregator.named_parameters, kind)
        if isinstance(plan, str):
            reason, plan = plan, None
    if reason is not None:
        return None, f"server agent ({reason})"
    cls = FlatFedAsync if kind == "async" else FlatFedAvg
    return cls(server, init_model, executor, *plan), f"flat buffer ({cls.__name__})"


def _is(obj, module: str, name: str) -> bool:
    """Whether ``obj``'s class is ``module.name``, without importing the module."""
    cls = type(obj)
    return cls.__module__ == module and cls.__qualname__ == name


def _unsupported_server(server, kind: str) -> Optional[str]:
    """Return why the flat path cannot stand in for ``server``'s own, if it cannot."""
    aggregator = getattr(server, "aggregator", None)
    scheduler = getattr(server, "scheduler", None)
    if kind == "async":
        expected = (
            ("appfl.algorithm.aggregator.fedasync_aggregator", "FedAsyncAggregator"),
            ("appfl.algorithm.scheduler.async_scheduler", "AsyncScheduler"),
        )
    else:
        expected = (
            ("appfl.algorithm.aggregator.fedavg_aggregator", "FedAvgAggregator"),
            ("appfl.algorithm.scheduler.sync_scheduler", "SyncScheduler"),
        )
    if not (_is(aggregator, *expected[0]) and _is(scheduler, *expected[1])):
        return (
            f"{type(aggregator).__name__} with {type(scheduler).__name__} is not "
            f"{expected[0][1]} with {expected[1][1]}"
        )
    base = next(
        (
            cls
            for cls in type(server).__mro__
            if cls.__module__ == "appfl.agent.server" and cls.__name__ == "ServerAgent"
        ),
        None,
    )
    if base is None or any(
        getattr(type(server), name) is not getattr(base, name)
        for name in ("global_update", "get_parameters")
    ):
        return "the server agent overrides global_update or get_parameters"
    if kind == "sync":
        configs = aggregator.aggregator_configs
        if configs.get("use_secure_agg", False):
            return "secure aggregation is on"
        if getattr(aggregator, "model_chunk_size", None) is not None:
            return "chunked aggregation is on"
    return None


def _plan_layout(init_model, named_parameters, kind: str):
    """Return (layout, regions) for the global model, or why there is none."""
    import torch

    groups: Dict[str, List[str]] = {group: [] for group in _GROUPS}
    for name, value in init_model.items():
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
            return f"{name!r} is not a dense tensor"
        if value.dtype in (torch.int64, torch.int32):
            group = "int64" if value.dtype == torch.int64 else "int32"
        elif named_parameters is None or name in named_parameters:
            if value.dtype != torch.float32:
                return f"parameter {name!r} is {value.dtype}, not float32"
            group = "param"
        elif value.dtype == torch.float32:
            group = "float"
        elif kind == "sync":
            return f"buffer {name!r} is {value.dtype}; FedAvg averages it"
        else:
            group = "other"
        groups[group].append(name)
    layout = make_layout(init_model, [name for group in _GROUPS for name in groups[group]])
    where = {name: (start, nbytes) for name, _, _, start, nbytes in layout}
    regions = {}
    for group, names in groups.items():
        if names and group in _DTYPES:
            last_start, last_nbytes = where[names[-1]]
            regions[group] = (where[names[0]][0], last_start + last_nbytes)
    return layout, regions


class _FlatGlobal:
    """The packed global model and what the two aggregations share."""

    def __init__(self, server, init_model, executor, layout, regions):
        self.server = server
        self.aggregator = server.aggregator
        self.scheduler = server.scheduler
        self.executor = executor
        self.regions: Dict[str, Tuple[int, int]] = regions
        self.schema, self.buffer = pack_state(init_model, layout=layout)
        self.nbytes = int(self.buffer.numel())
        self.updates = 0
        self._packed_sends = executor is not None and getattr(
            executor, "accepts_packed", False
        )

    def params(self):
        """The current global model, packed for the executor if it accepts that."""
        if self._packed_sends:
            return PackedState(self.schema, self.buffer)
        return self.state_dict()

    def state_dict(self) -> Dict:
        """Zero-copy, read-only views of the current global model."""
        return unpack_state(self.schema, self.buffer)

    def sync_server(self) -> None:
        """Leave the server's aggregator and model as APPFL's own path would."""
        if not self.updates:
            return  # nothing aggregated: the server still holds the initial model
        state = {name: t.clone() for name, t in self.state_dict().items()}
        self.aggregator.global_state = state
        if self.aggregator.model is not None:
            self.aggregator.model.load_state_dict(state, strict=False)

    def _local(self, local) -> Optional[PackedState]:
        """``local`` packed in the global model's layout, or None if it differs."""
        if not isinstance(local, PackedState):
            try:
                local = PackedState(*pack_state(local, layout=self.schema))
            except TypeError:
                return None
        return local if same_layout(local.schema, self.schema) else None

    def _view(self, buffer, group: str):
        """``group``'s region of ``buffer``, as one flat tensor of its dtype."""
        import torch

        start, end = self.regions[group]
        return buffer[start:end].view(getattr(torch, _DTYPES[group]))

    def _recycle(self, buffer) -> None:
        if self.executor is not None:
            self.executor.recycle(buffer)

    def _count_epoch(self) -> None:
        """Advance the scheduler's epoch count, as its ``schedule`` does."""
        with self.scheduler._access_lock:
            self.scheduler._num_global_epochs += 1


class FlatFedAsync(_FlatGlobal):
    """FedAsyncAggregator's update, applied to packed buffers."""

    def __init__(self, server, init_model, executor, layout, regions):
        super().__init__(server, init_model, executor, layout, regions)
        self.gradient_based = bool(
            self.aggregator.aggregator_configs.get("gradient_based", False)
        )

    def dispatch(self, cid: str):
        """Record ``cid``'s download, as ``get_parameters(client_id=cid)`` does."""
        self.aggregator.client_step[cid] = self.aggregator.global_step
        return self.params()

    def update(self, cid: str, local) -> bool:
        """
        Aggregate ``cid``'s local model into a new global version.

        :param local: The local model, packed or as a state dict. A packed buffer is
            overwritten: it becomes the new global model.
        :return: False, with nothing changed, if ``local`` does not have the global
            model's tensors; the caller must then use the server agent.
        """
        local = self._local(local)
        if local is None:
            return False
        if self.server.training_finished():  # global_update ignores the model
            self._recycle(local.buffer)
            return True
        alpha_t = self._alpha(cid)
        if "param" in self.regions:
            lp = self._view(local.buffer, "param")
            gp = self._view(self.buffer, "param")
            if self.gradient_based:
                lp.mul_(-alpha_t)
            else:
                lp.sub_(gp)
                lp.mul_(alpha_t)
            lp.add_(gp)
        agg = self.aggregator
        agg.global_step += 1
        agg.client_step[cid] = agg.global_step
        self._count_epoch()
        if not self.updates:
            # FedAsyncAggregator keys its global model by the first local model.
            self.schema = local.schema
        self.updates += 1
        previous, self.buffer = self.buffer, local.buffer
        self._recycle(previous)
        return True

    def _alpha(self, cid: str) -> float:
        """The mixing weight, exactly as ``FedAsyncAggregator.compute_steps``."""
        agg = self.aggregator
        if cid not in agg.client_step:
            warn = getattr(agg, "_warn_unrecorded_download", None)
            if warn is not None:
                warn(cid)
            agg.client_step[cid] = 0
        sizes = getattr(agg, "client_sample_size", None)
        if agg.client_weights_mode == "sample_size" and sizes and cid in sizes:
            weight = sizes[cid] / sum(sizes.values())
        else:
            weight = 1.0 / agg.aggregator_configs.get("num_clients", 1)
        staleness = agg.global_step - agg.client_step[cid]
        return agg.alpha * agg.staleness_fn(staleness) * weight


class FlatFedAvg(_FlatGlobal):
    """FedAvgAggregator's round, summed into a packed buffer as models arrive."""

    def __init__(self, server, init_model, executor, layout, regions):
        super().__init__(server, init_model, executor, layout, regions)
        self._round: Optional[Dict] = None

    def begin_round(self, cids: List[str]) -> None:
        """Start a round that aggregates the local models of ``cids``, in order."""
        if len(set(cids)) != len(cids):
            raise ValueError("a round cannot aggregate a client twice")
        import torch

        agg = self.aggregator
        sizes = getattr(agg, "client_sample_size", None) or {}
        total = sum(sizes[cid] for cid in cids if cid in sizes)
        weights = {}
        for cid in cids:  # as FedAvgAggregator.compute_steps
            if agg.client_weights_mode == "sample_size" and cid in sizes and total > 0:
                weights[cid] = sizes[cid] / total
            else:
                weights[cid] = 1.0 / len(cids)
        self._round = {
            "order": list(cids),
            "weights": weights,
            "sum": torch.zeros(self.nbytes, dtype=torch.uint8),
            "schema": None,
            "added": 0,
        }

    @property
    def round_started(self) -> bool:
        """Whether a round has begun and taken in at least one model."""
        return self._round is not None and self._round["added"] > 0

    def add(self, cid: str, local) -> bool:
        """
        Add the round's next local model, in the order given to :meth:`begin_round`.

        :param local: The local model, packed or as a state dict. A packed buffer is
            overwritten and then recycled.
        :return: False, with nothing changed, if ``local`` does not have the global
            model's tensors.
        """
        state = self._round
        if state is None or state["added"] >= len(state["order"]):
            raise RuntimeError("add() needs a round with a model still to come")
        if state["order"][state["added"]] != cid:
            raise RuntimeError(
                f"expected {state['order'][state['added']]}'s model next, got {cid}'s"
            )
        local = self._local(local)
        if local is None:
            return False
        weight = state["weights"][cid]
        for group in self.regions:
            lp = self._view(local.buffer, group)
            if group == "param":
                lp.sub_(self._view(self.buffer, group))
                lp.mul_(weight)
            self._view(state["sum"], group).add_(lp)
        if state["schema"] is None:
            state["schema"] = local.schema
        state["added"] += 1
        self._recycle(local.buffer)
        return True

    def finish(self) -> None:
        """Turn the round's sum into the next global version."""
        import torch

        state, self._round = self._round, None
        if state is None or state["added"] != len(state["order"]):
            raise RuntimeError("finish() needs every model of the round")
        count = len(state["order"])
        total = state["sum"]
        for group in self.regions:
            sp = self._view(total, group)
            if group == "param":
                sp.add_(self._view(self.buffer, group))
            elif group == "float":
                sp.div_(count)
            else:  # integers: true division, then back to the integer type
                sp.copy_(torch.div(sp, count).type(sp.dtype))
        self._count_epoch()
        if not self.updates:
            # FedAvgAggregator keys its global model by the first local model.
            self.schema = state["schema"]
        self.updates += 1
        previous, self.buffer = self.buffer, total
        self._recycle(previous)
