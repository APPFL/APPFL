import copy
import torch
from omegaconf import DictConfig
from appfl.aggregator import BaseAggregator
from typing import Union, Dict, OrderedDict, Any

class HFLRootFedAvgAggregator(BaseAggregator):
    def __init__(
        self,
        model: torch.nn.Module,
        aggregator_config: DictConfig,
        logger: Any
    ):
        self.model = model
        self.logger = logger
        self.aggregator_config = aggregator_config

        self.named_parameters = set()
        for name, _ in self.model.named_parameters():
            self.named_parameters.add(name)

        self.step = {}

    def get_parameters(self, **kwargs) -> Dict:
        return copy.deepcopy(self.model.state_dict())

    def aggregate(self, local_models: Dict[Union[str, int], Union[Dict, OrderedDict]], **kwargs) -> Dict:
        """
        Take the weighted average of local models from clients and return the global model.
        """
        global_state = copy.deepcopy(self.model.state_dict())
        
        num_clients_dict = kwargs.get("num_clients", {})
        if not hasattr(self, "total_num_clients"):
            self.total_num_clients = 0
            for client_id in local_models:
                self.total_num_clients += num_clients_dict.get(client_id, 1)
        
        self.compute_steps(local_models, num_clients_dict)
        
        for name in self.model.state_dict():
            if name not in self.named_parameters:
                param_sum = torch.zeros_like(self.model.state_dict()[name])
                for client_id, model in local_models.items():
                    param_sum += model[name] * num_clients_dict.get(client_id, 1)
                global_state[name] = torch.div(param_sum, self.total_num_clients)
            else:
                global_state[name] += self.step[name]
            
        self.model.load_state_dict(global_state)
        return global_state
    
    def compute_steps(
        self, 
        local_models: Dict[Union[str, int], Union[Dict, OrderedDict]],
        num_clients_dict: Dict[Union[str, int], int]
    ):
        """
        Compute the changes to the global model after the aggregation.
        """
        for name in self.named_parameters:
            self.step[name] = torch.zeros_like(self.model.state_dict()[name])
            
        for client_id, model in local_models.items():
            weight = (1.0 / self.total_num_clients) * num_clients_dict.get(client_id, 1)
            for name in self.named_parameters:
                self.step[name] += weight * (model[name] - self.model.state_dict()[name])
