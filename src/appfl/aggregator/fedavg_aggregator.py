import copy
import torch
from omegaconf import DictConfig
from appfl.aggregator import BaseAggregator
from typing import Union, Dict, OrderedDict

class FedAvgAggregator(BaseAggregator):
    def __init__(
        self,
        model: torch.nn.Module,
        client_weights: DictConfig,
        aggregator_config: DictConfig,
    ):
        self.model = model
        self.client_weights = client_weights
        self.aggregator_config = aggregator_config

        self.named_parameters = set()
        for name, _ in self.model.named_parameters():
            self.named_parameters.add(name)

    def aggregate(self, local_models: Dict[Union[str, int], Union[Dict, OrderedDict]]) -> Dict:
        """
        Take the weighted average of local models from clients and return the global model.
        """
        # Initialize the global model to zeros
        for name in self.model.state_dict():
            self.model.state_dict()[name] = torch.zeros_like(self.model.state_dict()[name])

        for client_id, model in local_models.items():
            for name in self.model.state_dict():
                if name in self.named_parameters:
                    self.model.state_dict()[name] += self.client_weights[client_id] * model[name]
                else:
                    self.model.state_dict()[name] += model[name]
        for name in self.model.state_dict():
            if name not in self.named_parameters:
                self.model.state_dict()[name] = torch.div(self.model.state_dict()[name], len(local_models))
        return copy.deepcopy(self.model.state_dict())
    
    def get_parameters(self) -> Union[Dict, OrderedDict]:
        return copy.deepcopy(self.model.state_dict())
