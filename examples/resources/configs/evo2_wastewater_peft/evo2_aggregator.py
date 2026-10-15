"""
Server-side APPFL aggregator for the Evo2 wastewater-PEFT federated case study.
Loaded via `aggregator_path` (server_fedavg.yaml). Pairs with `evo2_trainer.py`.

No `model_configs` is set server-side, so `self.model` is always `None` here —
`get_parameters()` falls back to the aggregated adapter state instead. Clients
whose data failed the AIDRIN/DRISEE gate send a sentinel instead of real
weights; this aggregator filters those out before FedAvg runs.
"""

from copy import deepcopy
from typing import Any, Dict, Optional, OrderedDict, Union

from omegaconf import DictConfig

from appfl.algorithm.aggregator import FedAvgAggregator

#: Key a client sends instead of real parameters when its local data failed
#: the AIDRIN/DRISEE readiness gate. Must match `AIDRIN_SKIP_KEY` in
#: `aidrin_checker.py` / `evo2_trainer.py` on the client side.
AIDRIN_SKIP_KEY = "__aidrin_skip__"


class Evo2FedAvgAggregator(FedAvgAggregator):
    """FedAvg aggregator for a model-less global state, with an AIDRIN/DRISEE
    contamination filter applied before aggregation."""

    def __init__(
        self,
        model: Optional[Any] = None,
        aggregator_configs: DictConfig = DictConfig({}),
        logger: Optional[Any] = None,
    ):
        super().__init__(
            model=model, aggregator_configs=aggregator_configs, logger=logger
        )
        self.aidrin_skip_key = aggregator_configs.get(
            "aidrin_skip_key", AIDRIN_SKIP_KEY
        )

    def _log(self, msg: str) -> None:
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def get_parameters(self, **kwargs) -> Union[Dict, OrderedDict]:
        """
        Return the last aggregated adapter state. Falls back to this instead
        of `self.model.state_dict()` (the usual `FedAvgAggregator` behavior)
        because no model is attached.
        """
        if self.model is None or not self.model.state_dict():
            if self.global_state is not None:
                return deepcopy(self.global_state)
            return OrderedDict()
        return super().get_parameters(**kwargs)

    def aggregate(
        self,
        local_models: Dict[Union[str, int], Union[Dict, OrderedDict]],
        **kwargs,
    ) -> Dict:
        """Drop AIDRIN-contaminated clients for this round, then delegate to FedAvg."""
        contaminated = [
            client_id
            for client_id, params in local_models.items()
            if self.aidrin_skip_key in params
        ]
        if contaminated:
            self._log(
                f"[Evo2FedAvgAggregator] AIDRIN: skipping contaminated clients: {contaminated}"
            )
            local_models = {
                client_id: params
                for client_id, params in local_models.items()
                if client_id not in contaminated
            }

        if not local_models:
            self._log(
                "[Evo2FedAvgAggregator] AIDRIN: all clients contaminated this round "
                "— global state unchanged"
            )
            if self.global_state is not None:
                return deepcopy(self.global_state)
            return OrderedDict()

        # First round: seed global_state from a client's own keys when they
        # don't match the (nonexistent) model, so FedAvg has something to
        # average into from round 1 instead of raising on a missing model.
        if self.global_state is None and self.model is not None:
            model_state = self.model.state_dict()
            first_client_params = next(iter(local_models.values()))
            if not all(name in model_state for name in first_client_params):
                self.global_state = {
                    name: (
                        model_state[name].clone().detach()
                        if name in model_state
                        else first_client_params[name].detach().clone()
                    )
                    for name in first_client_params
                }

        return super().aggregate(local_models, **kwargs)
