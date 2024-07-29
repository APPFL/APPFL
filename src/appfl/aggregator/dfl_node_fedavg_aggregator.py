import copy
import torch
from omegaconf import DictConfig
from appfl.aggregator import BaseAggregator
from typing import Union, Dict, OrderedDict, Any, List

class DFLNodeFedAvgAggregator(BaseAggregator):
    def __init__(
        self,
        model: torch.nn.Module,
        aggregator_config: DictConfig,
        logger: Any
    ):
        self.model = model
        self.logger = logger
        self.aggregator_config = aggregator_config

    def get_parameters(self, **kwargs) -> Dict:
        return copy.deepcopy(self.model.state_dict())
    
    def aggregate(
        self,
        local_model: Union[Dict, OrderedDict],
        neighbor_models: Union[Dict[Union[str, int], Union[Dict, OrderedDict]], List[Union[Dict, OrderedDict]]],
        **kwargs,
    ):
        new_model = copy.deepcopy(self.model.state_dict())
        for name in self.model.state_dict():
            param_sum = torch.zeros_like(self.model.state_dict()[name])
            for model in neighbor_models:
                param_sum += model[name]
            param_sum += local_model[name]
            new_model[name] = torch.div(param_sum, len(neighbor_models) + 1)
        
        self.model.load_state_dict(new_model)
        return new_model
    