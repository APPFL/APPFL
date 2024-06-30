import copy
import torch
from omegaconf import DictConfig
from appfl.aggregator import BaseAggregator
from typing import Union, Dict, OrderedDict, Any

class FedAsyncAggregator(BaseAggregator):
    """
    FedAsync Aggregator class for Federated Learning.
    For more details, check paper: https://arxiv.org/pdf/1903.03934.pdf
    """
    def __init__(
        self,
        model: torch.nn.Module,
        aggregator_config: DictConfig,
        logger: Any
    ):
        self.model = model
        self.client_weights_mode = aggregator_config.get("client_weights_mode", "equal")
        self.aggregator_config = aggregator_config
        self.logger = logger

        self.named_parameters = set()
        for name, _ in self.model.named_parameters():
            self.named_parameters.add(name)
        self.staleness_fn = self.__staleness_fn_factory(
            staleness_fn_name= self.aggregator_config.get("staleness_fn", "constant"),
            **self.aggregator_config.get("staleness_fn_kwargs", {})
        )
        self.alpha = self.aggregator_config.get("alpha", 0.9)
        self.global_step = 0
        self.client_step = {}
        self.step = {}

    def get_parameters(self, **kwargs) -> Dict:
        return copy.deepcopy(self.model.state_dict())

    def aggregate(self, client_id: Union[str, int], local_model: Union[Dict, OrderedDict], **kwargs) -> Dict:
        global_state = copy.deepcopy(self.model.state_dict())
        
        self.compute_steps(client_id, local_model)
        
        for name in self.model.state_dict():
            if name not in self.named_parameters:
                global_state[name] = local_model[name]
            else:
                global_state[name] += self.step[name]
        self.model.load_state_dict(global_state)
        self.global_step += 1
        self.client_step[client_id] = self.global_step
        return global_state
    
    def compute_steps(self, client_id: Union[str, int], local_model: Union[Dict, OrderedDict],):
        """
        Compute changes to the global model after the aggregation.
        """
        if client_id not in self.client_step:
            self.client_step[client_id] = 0
        gradient_based = self.aggregator_config.get("gradient_based", False)
        if (
            self.client_weights_mode == "sample_size" and
            hasattr(self, "client_sample_size") and
            client_id in self.client_sample_size
        ):
            weight = self.client_sample_size[client_id] / sum(self.client_sample_size.values())
        else:
            weight = 1.0 / self.aggregator_config.get("num_clients", 1)
        alpha_t = self.alpha * self.staleness_fn(self.global_step - self.client_step[client_id]) * weight
        for name in self.named_parameters:
            self.step[name] = (
                alpha_t * (-local_model[name]) if gradient_based
                else alpha_t * (local_model[name] - self.model.state_dict()[name])
            )

    def __staleness_fn_factory(self, staleness_fn_name, **kwargs):
        if staleness_fn_name   == "constant":
            return lambda u : 1
        elif staleness_fn_name == "polynomial":
            a = kwargs['a']
            return lambda u:  (u + 1) ** a
        elif staleness_fn_name == "hinge":
            a = kwargs['a']
            b = kwargs['b']
            return lambda u: 1 if u <= b else 1.0/ (a * (u - b) + 1.0)
        else:
            raise NotImplementedError
