import torch
from omegaconf import DictConfig
from appfl.aggregator import FedAvgAggregator
from typing import Union, Dict, OrderedDict, Any

class FedAdagradAggregator(FedAvgAggregator):
    """
    FedAdagrad Aggregator class for Federated Learning.
    For more details, check paper `Adaptive Federated Optimization`
    at https://arxiv.org/pdf/2003.00295.pdf

    Required aggregator_config fields:
        - server_learning_rate: `eta` in the paper
        - server_adapt_param: `tau` in the paper
        - server_momentum_param_1: `beta_1` in the paper
    """
    def __init__(
        self,
        model: torch.nn.Module,
        aggregator_config: DictConfig,
        logger: Any
    ):
        super().__init__(model, aggregator_config, logger)
        self.m_vector = {}
        self.v_vector = {}
        for name in self.named_parameters:
            self.m_vector[name] = torch.zeros_like(self.model.state_dict()[name])
            self.v_vector[name] = torch.zeros_like(self.model.state_dict()[name]) + self.aggregator_config.server_adapt_param**2
    
    def compute_steps(self, local_models: Dict[Union[str, int], Union[Dict, OrderedDict]]):
        """
        Compute the changes to the global model after the aggregation.
        """
        super().compute_steps(local_models)
        for name in self.named_parameters:
            self.m_vector[name] = (
                self.aggregator_config.server_momentum_param_1 * self.m_vector[name] + 
                (1-self.aggregator_config.server_momentum_param_1) * self.step[name]
            )
            self.v_vector[name] += torch.square(self.step[name])
            self.step[name] = torch.div(
                self.aggregator_config.server_learning_rate * self.m_vector[name],
                torch.sqrt(self.v_vector[name]) + self.aggregator_config.server_adapt_param
            )
