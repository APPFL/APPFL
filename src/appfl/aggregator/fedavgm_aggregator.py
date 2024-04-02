import torch
from omegaconf import DictConfig
from appfl.aggregator import FedAvgAggregator
from typing import Union, Dict, OrderedDict, Any

class FedAvgMAggregator(FedAvgAggregator):
    """
    FedAvgM Aggregator class for Federated Learning.
    For more details, check paper `Measuring the effects of non-identical data distribution for federated visual classification`
    at https://arxiv.org/pdf/1909.06335.pdf

    Required aggregator_config fields:
        - server_momentum_param_1: `beta` in the paper
    """
    def __init__(
        self,
        model: torch.nn.Module,
        aggregator_config: DictConfig,
        logger: Any
    ):
        super().__init__(model, aggregator_config, logger)
        self.v_vector = {}
        for name in self.named_parameters:
            self.v_vector[name] = torch.zeros_like(self.model.state_dict()[name])
    
    def compute_steps(self, local_models: Dict[Union[str, int], Union[Dict, OrderedDict]]):
        """
        Compute the changes to the global model after the aggregation.
        """
        super().compute_steps(local_models)
        for name in self.named_parameters:
            self.v_vector[name] = self.aggregator_config.server_momentum_param_1 * self.v_vector[name] + self.step[name]
            self.step[name] = self.v_vector[name]
