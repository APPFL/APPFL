import torch
from omegaconf import DictConfig
from appfl.algorithm.aggregator import FedAvgAggregator
from typing import Union, Dict, OrderedDict, Any, Optional


class FedYogiAggregator(FedAvgAggregator):
    """
    FedYogi Aggregator class for Federated Learning.
    For more details, check paper `Adaptive Federated Optimization`
    at https://arxiv.org/pdf/2003.00295.pdf

    Required aggregator_configs fields:
        - server_learning_rate: `eta` in the paper
        - server_adapt_param: `tau` in the paper
        - server_momentum_param_1: `beta_1` in the paper
        - server_momentum_param_2: `beta_2` in the paper
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        aggregator_configs: DictConfig = DictConfig({}),
        logger: Optional[Any] = None,
    ):
        super().__init__(model, aggregator_configs, logger)
        self.m_vector = {}
        self.v_vector = {}

    def compute_steps(
        self, local_models: Dict[Union[str, int], Union[Dict, OrderedDict]]
    ):
        """
        Compute the changes to the global model after the aggregation.
        """
        super().compute_steps(local_models)
        if len(self.m_vector) == 0:
            for name in self.step:
                self.m_vector[name] = torch.zeros_like(self.step[name])
                self.v_vector[name] = (
                    torch.zeros_like(self.step[name])
                    + self.aggregator_configs.server_adapt_param**2
                )

        for name in self.step:
            self.m_vector[name] = (
                self.aggregator_configs.server_momentum_param_1 * self.m_vector[name]
                + (1 - self.aggregator_configs.server_momentum_param_1)
                * self.step[name]
            )
            self.v_vector[name] -= (
                1 - self.aggregator_configs.server_momentum_param_2
            ) * torch.mul(
                torch.square(self.step[name]),
                torch.sign(self.v_vector[name] - torch.square(self.step[name])),
            )
            self.step[name] = torch.div(
                self.aggregator_configs.server_learning_rate * self.m_vector[name],
                torch.sqrt(self.v_vector[name])
                + self.aggregator_configs.server_adapt_param,
            )
