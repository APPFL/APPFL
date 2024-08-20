import copy
import torch
from omegaconf import DictConfig
from appfl.aggregator import BaseAggregator
from typing import Union, Dict, OrderedDict, Any, Optional

class FedCompassAggregator(BaseAggregator):
    """
    FedCompass asynchronous federated learning algorithm.
    For more details, check paper: https://arxiv.org/abs/2309.14675
    """
    def __init__(
        self,
        model: torch.nn.Module,
        aggregator_config: DictConfig,
        logger: Any
    ):
        self.model = model
        self.logger = logger
        self.aggregator_config = aggregator_config
        self.staleness_fn = self.__staleness_fn_factory(
            staleness_fn_name= self.aggregator_config.get("staleness_fn", "constant"),
            **self.aggregator_config.get("staleness_fn_kwargs", {})
        )
        self.alpha = self.aggregator_config.get("alpha", 0.9)

        self.named_parameters = set()
        for name, _ in self.model.named_parameters():
            self.named_parameters.add(name)

    def aggregate(
            self,
            client_id: Optional[Union[str, int]]=None,
            local_model: Optional[Union[Dict, OrderedDict]] = None,
            local_models: Optional[Dict[Union[str, int], Union[Dict, OrderedDict]]] = None,
            staleness: Optional[Union[int, Dict[Union[str, int], int]]] = None,
            **kwargs
        ) -> Dict:
        global_state = copy.deepcopy(self.model.state_dict())
        # returned_global_state = {}
        gradient_based = self.aggregator_config.get("gradient_based", False)
        if client_id is not None and local_model is not None:
            self.logger.info(f"Updating Global Model from {client_id} with staleness {staleness}")
            weight = 1.0 / self.aggregator_config.get("num_clients", 1)
            alpha_t = self.alpha * self.staleness_fn(staleness) * weight
            for name in self.model.state_dict():
                if name in self.named_parameters:
                    if gradient_based:
                        global_state[name] -= local_model[name] * alpha_t
                    else:
                        global_state[name] -= (global_state[name] - local_model[name]) * alpha_t
                else:
                    global_state[name] = local_model[name]
        else:
            client_ids = []
            for client_id in local_models:
                client_ids.append(client_id)
            self.logger.info(f"Updating Global Model from {client_ids} with staleness {staleness}")
            for i, client_id in enumerate(local_models):
                local_model = local_models[client_id]
                weight = 1.0 / self.aggregator_config.get("num_clients", 1)
                self.logger.info(f"Weight for {client_id} is {weight}")
                alpha_t = self.alpha * self.staleness_fn(staleness[client_id]) * weight
                self.logger.info(f"Alpha for {client_id} is {alpha_t}")
                for name in self.model.state_dict():
                    if name in self.named_parameters:
                        if gradient_based:
                            global_state[name] -= local_model[name] * alpha_t
                        else:
                            global_state[name] -= (self.model.state_dict()[name] - local_model[name]) * alpha_t
                    else:
                        if i == 0:
                            global_state[name] = torch.zeros_like(self.model.state_dict()[name])
                        global_state[name] += local_model[name]
                        if i == len(local_models) - 1:
                            global_state[name] = torch.div(global_state[name], len(local_models))
        self.model.load_state_dict(global_state)
        return global_state
        # for name in self.named_parameters:
        #     returned_global_state[name] = global_state[name]
        
        # return returned_global_state

    def get_parameters(self, **kwargs) -> Dict:
        return copy.deepcopy(self.model.state_dict())

    def __staleness_fn_factory(self, staleness_fn_name, **kwargs):
        if staleness_fn_name   == "constant":
            return lambda u : 1
        elif staleness_fn_name == "polynomial":
            a = kwargs['a']
            return lambda u:  (u + 1) ** (-a)
        elif staleness_fn_name == "hinge":
            a = kwargs['a']
            b = kwargs['b']
            return lambda u: 1 if u <= b else 1.0/ (a * (u - b) + 1.0)
        else:
            raise NotImplementedError