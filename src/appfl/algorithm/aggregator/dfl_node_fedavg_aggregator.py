import copy
import torch
from omegaconf import DictConfig
from appfl.algorithm.aggregator import BaseAggregator
from typing import Any, Dict, List, Optional, OrderedDict, Tuple, Union


class DFLNodeFedAvgAggregator(BaseAggregator):
    """
    Average a DFL node's own model with its neighbors' models.

    This is the decentralized counterpart of :class:`FedAvgAggregator`. The difference is
    whose models are being averaged: `FedAvgAggregator` runs once per round on a server and
    sees every client, while this runs on every node and sees only that node's closed
    neighborhood -- itself plus its neighbors.

    The default weighting is uniform over that neighborhood, ``1 / (|N_i| + 1)``, which is the
    standard mixing matrix for decentralized averaging. Pass ``mixing_weights`` to override,
    for an irregular graph where uniform weights over a neighborhood do not give a doubly
    stochastic mixing matrix.

    :param `model`: the model being trained, used as the template for parameter names.
    :param `aggregator_configs`: configuration, from `aggregator_kwargs` in the YAML.
    :param `logger`: an optional logger.
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        aggregator_configs: DictConfig = DictConfig({}),
        logger: Optional[Any] = None,
    ):
        self.model = model
        self.logger = logger
        self.aggregator_configs = aggregator_configs

    def get_parameters(self, **kwargs) -> Dict:
        if self.model is None:
            raise ValueError(
                "DFLNodeFedAvgAggregator was constructed without a model, so it has no "
                "parameters of its own to return; read them from the node's trainer instead."
            )
        return copy.deepcopy(self.model.state_dict())

    def aggregate(
        self,
        local_model: Union[Dict, OrderedDict],
        neighbor_models: Union[
            Dict[Union[str, int], Union[Dict, OrderedDict]],
            List[Union[Dict, OrderedDict]],
        ],
        local_id: str,
        mixing_weights: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> Dict:
        """Return the mixed model: this node's parameters averaged with its neighbors'.

        :param local_model: this node's own parameters after local training.
        :param neighbor_models: the neighbors' parameters, keyed by node id or as a list.
        :param local_id: this node's id. Required: it is how the node finds its own weight,
            and every caller knows it, so leaving it optional only bought a branch that had to
            guess which entry in `mixing_weights` belonged to the node.
        :param mixing_weights: optional ``{node_id: weight}``, including this node's own id.
            Ignored when `neighbor_models` is a list, since there is then no id to key on.
        """
        named = isinstance(neighbor_models, dict)
        models = list(neighbor_models.values()) if named else list(neighbor_models)
        if not models:
            return dict(local_model)

        own_weight, peer_weights = self._resolve_weights(
            neighbor_models if named else None, mixing_weights, local_id, len(models)
        )

        new_model = {}
        for name in local_model:
            mixed = local_model[name] * own_weight
            if named:
                for node_id, model in neighbor_models.items():
                    mixed = mixed + model[name] * peer_weights[str(node_id)]
            else:
                for i, model in enumerate(models):
                    mixed = mixed + model[name] * peer_weights[str(i)]
            # Integer buffers (e.g. `num_batches_tracked`) must not silently become floats.
            new_model[name] = (
                mixed.to(local_model[name].dtype)
                if torch.is_tensor(local_model[name])
                else mixed
            )

        if self.model is not None:
            self.model.load_state_dict(new_model)
        return new_model

    def _resolve_weights(
        self,
        neighbor_models: Optional[Dict[Union[str, int], Any]],
        mixing_weights: Optional[Dict[str, float]],
        local_id: str,
        n_neighbors: int,
    ) -> Tuple[float, Dict[str, float]]:
        """Return ``(own_weight, {peer_key: weight})``, normalized to sum to one."""
        if mixing_weights is None or neighbor_models is None:
            uniform = 1.0 / (n_neighbors + 1)
            keys = (
                [str(k) for k in neighbor_models]
                if neighbor_models is not None
                else [str(i) for i in range(n_neighbors)]
            )
            return uniform, {key: uniform for key in keys}

        peers = {
            str(node_id): float(mixing_weights[str(node_id)])
            for node_id in neighbor_models
        }

        own = (
            float(mixing_weights[str(local_id)])
            if str(local_id) in mixing_weights
            else 1.0 - sum(peers.values())
        )

        total = own + sum(peers.values())
        if total <= 0:
            raise ValueError(
                f"mixing weights sum to {total}, which cannot be normalized"
            )
        return own / total, {key: value / total for key, value in peers.items()}
