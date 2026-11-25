"""Equation (7): sample-weighted, work-normalized synchronous aggregation."""

import math

import torch
from appfl.algorithm.aggregator.base_aggregator import BaseAggregator


class TCSAggregator(BaseAggregator):
    """Use with SyncScheduler. num_steps / round / learning_rate are client metadata.

    Floating buffers use sample-weighted averaging; nonfloating buffers must
    agree across clients. Only trainable parameters use work normalization.
    """

    def __init__(self, model=None, aggregator_configs=None, logger=None):
        if model is None:
            raise ValueError("TCSAggregator requires the initial global model")
        self.model = model
        self.logger = logger
        self.aggregator_configs = aggregator_configs or {}
        self.names = {n for n, p in model.named_parameters() if p.requires_grad}
        self.state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
        self.round = 0

    def get_parameters(self, **kwargs):
        return {k: v.clone() for k, v in self.state.items()}, {"round": self.round}

    def aggregate(self, local_models, **kwargs):
        if not local_models:
            raise ValueError("no client updates")
        ids = list(local_models)
        steps, sizes, rates = {}, {}, []
        for cid in ids:
            h = kwargs.get("num_steps", {}).get(cid)
            n = kwargs.get("num_samples", {}).get(cid)
            r = kwargs.get("round", {}).get(cid)
            lr = kwargs.get("learning_rate", {}).get(cid)
            if isinstance(h, bool) or not isinstance(h, int) or h < 1:
                raise ValueError("each update needs a positive integer num_steps")
            if isinstance(n, bool) or not isinstance(n, int) or n < 1:
                raise ValueError("each update needs a positive integer num_samples")
            if isinstance(r, bool) or not isinstance(r, int) or r != self.round:
                raise ValueError("stale or missing global round")
            if (
                not isinstance(lr, (int, float))
                or isinstance(lr, bool)
                or not math.isfinite(lr)
                or lr <= 0
            ):
                raise ValueError("each update needs a finite positive learning_rate")
            registered = getattr(self, "client_sample_size", {}).get(cid, n)
            if registered != n:
                raise ValueError("reported sample size differs from registered size")
            if set(local_models[cid]) != set(self.state):
                raise ValueError("model keys differ from the global model")
            for k, v in local_models[cid].items():
                if (
                    v.shape != self.state[k].shape
                    or v.dtype != self.state[k].dtype
                    or not torch.isfinite(v).all()
                ):
                    raise ValueError(f"invalid tensor {k}")
            steps[cid], sizes[cid] = h, n
            rates.append(lr)
        if any(lr != rates[0] for lr in rates):
            raise ValueError("TCS requires a common learning rate within a round")
        weights = {cid: sizes[cid] / sum(sizes.values()) for cid in ids}
        mean_steps = sum(weights[cid] * steps[cid] for cid in ids)
        updated = {}
        with torch.no_grad():
            for key, previous in self.state.items():
                values = {cid: local_models[cid][key].detach().cpu() for cid in ids}
                if key in self.names:
                    delta = sum(
                        (values[cid] - previous) * (weights[cid] / steps[cid])
                        for cid in ids
                    )
                    updated[key] = previous + mean_steps * delta
                elif previous.is_floating_point():
                    updated[key] = sum(values[cid] * weights[cid] for cid in ids)
                else:
                    if any(not torch.equal(values[cid], values[ids[0]]) for cid in ids):
                        raise ValueError(
                            f"nonfloating buffer {key} differs across clients"
                        )
                    updated[key] = values[ids[0]].clone()
        if any(not torch.isfinite(v).all() for v in updated.values()):
            raise FloatingPointError("non-finite aggregate")
        self.model.load_state_dict(updated, strict=True)
        self.state = updated
        self.round += 1
        return self.get_parameters()
