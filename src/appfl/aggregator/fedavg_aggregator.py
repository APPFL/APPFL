import copy
import torch
from omegaconf import DictConfig
from appfl.aggregator import BaseAggregator
from typing import Union, Dict, OrderedDict, Any

class FedAvgAggregator(BaseAggregator):
    def __init__(
        self,
        model: torch.nn.Module,
        aggregator_config: DictConfig,
        logger: Any
    ):
        self.model = model
        self.client_weights = aggregator_config.get("client_weights", "equal")
        self.aggregator_config = aggregator_config
        self.logger = logger

        self.named_parameters = set()
        for name, _ in self.model.named_parameters():
            self.named_parameters.add(name)

    def aggregate(self, local_models: Dict[Union[str, int], Union[Dict, OrderedDict]], **kwargs) -> Dict:
        """
        Take the weighted average of local models from clients and return the global model.
        """
        global_state = copy.deepcopy(self.model.state_dict())
        for name in self.model.state_dict():
            global_state[name] = torch.zeros_like(self.model.state_dict()[name])
        
        for client_id, model in local_models.items():
            for name in self.model.state_dict():
                if name in self.named_parameters:
                    weight = self.client_weights[client_id] if isinstance(self.client_weights, dict) else 1.0 / len(local_models)
                    global_state[name] += weight * model[name]
                else:
                    global_state[name] += model[name]
        
        for name in self.model.state_dict():
            if name not in self.named_parameters:
                global_state[name] = torch.div(global_state[name], len(local_models))
        self.model.load_state_dict(global_state)
        return global_state
    
    def get_parameters(self, **kwargs) -> Dict:
        return copy.deepcopy(self.model.state_dict())
