import gc
from collections import OrderedDict
from typing import Any

import torch
from omegaconf import DictConfig

from appfl.algorithm.aggregator import FedAvgAggregator
from appfl.misc.memory_utils import optimize_memory_cleanup, safe_inplace_operation


class FedAdagradAggregator(FedAvgAggregator):
    """
    FedAdagrad Aggregator class for Federated Learning.
    For more details, check paper `Adaptive Federated Optimization`
    at https://arxiv.org/pdf/2003.00295.pdf

    Required aggregator_configs fields:
        - server_learning_rate: `eta` in the paper
        - server_adapt_param: `tau` in the paper
        - server_momentum_param_1: `beta_1` in the paper
    """

    def __init__(
        self,
        model: torch.nn.Module | None = None,
        aggregator_configs: DictConfig = DictConfig({}),
        logger: Any | None = None,
    ):
        super().__init__(model, aggregator_configs, logger)
        self.m_vector = {}
        self.v_vector = {}

    def compute_steps(self, local_models: dict[str | int, dict | OrderedDict]):
        """
        Compute the changes to the global model after the aggregation.
        """
        super().compute_steps(local_models)

        # Memory optimization: Initialize vectors efficiently
        if len(self.m_vector) == 0:
            if self.optimize_memory:
                with torch.no_grad():
                    for name in self.step:
                        self.m_vector[name] = torch.zeros_like(self.step[name])
                        self.v_vector[name] = (
                            torch.zeros_like(self.step[name])
                            + self.aggregator_configs.server_adapt_param**2
                        )
                    gc.collect()
            else:
                for name in self.step:
                    self.m_vector[name] = torch.zeros_like(self.step[name])
                    self.v_vector[name] = (
                        torch.zeros_like(self.step[name])
                        + self.aggregator_configs.server_adapt_param**2
                    )

        # Memory optimization: Use safe in-place operations
        if self.optimize_memory:
            with torch.no_grad():
                for name in self.step:
                    # Momentum update with safe operations
                    momentum_term = (
                        self.m_vector[name]
                        * self.aggregator_configs.server_momentum_param_1
                    )
                    step_term = self.step[name] * (
                        1 - self.aggregator_configs.server_momentum_param_1
                    )
                    self.m_vector[name] = safe_inplace_operation(
                        momentum_term, "add", step_term
                    )

                    # Variance update with safe operations
                    step_squared = torch.square(self.step[name])
                    self.v_vector[name] = safe_inplace_operation(
                        self.v_vector[name], "add", step_squared
                    )

                    # Final step computation with safe operations
                    numerator = (
                        self.aggregator_configs.server_learning_rate
                        * self.m_vector[name]
                    )
                    denominator = (
                        torch.sqrt(self.v_vector[name])
                        + self.aggregator_configs.server_adapt_param
                    )
                    self.step[name] = safe_inplace_operation(
                        numerator, "div", denominator
                    )

                    # Cleanup intermediate tensors
                    optimize_memory_cleanup(
                        momentum_term,
                        step_term,
                        step_squared,
                        numerator,
                        denominator,
                        force_gc=False,
                    )

                optimize_memory_cleanup(force_gc=True)
        else:
            # Original behavior
            for name in self.step:
                self.m_vector[name] = (
                    self.aggregator_configs.server_momentum_param_1
                    * self.m_vector[name]
                    + (1 - self.aggregator_configs.server_momentum_param_1)
                    * self.step[name]
                )
                self.v_vector[name] += torch.square(self.step[name])
                self.step[name] = torch.div(
                    self.aggregator_configs.server_learning_rate * self.m_vector[name],
                    torch.sqrt(self.v_vector[name])
                    + self.aggregator_configs.server_adapt_param,
                )
