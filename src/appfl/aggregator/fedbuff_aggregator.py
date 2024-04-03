import copy
import torch
from omegaconf import DictConfig
from appfl.aggregator import FedAsyncAggregator
from typing import Union, Dict, OrderedDict, Any

class FedBuffAggregator(FedAsyncAggregator):
    """
    FedBuff Aggregator class for Federated Learning.
    For more details, check paper: https://proceedings.mlr.press/v151/nguyen22b/nguyen22b.pdf
    """
    def __init__(
        self,
        model: torch.nn.Module,
        aggregator_config: DictConfig,
        logger: Any
    ):
        super().__init__(model, aggregator_config, logger)
        self.buff_size = 0
        self.K = self.aggregator_config.K

    def aggregate(self, client_id: Union[str, int], local_model: Union[Dict, OrderedDict], **kwargs) -> Dict:
        global_state = copy.deepcopy(self.model.state_dict())
        
        self.compute_steps(client_id, local_model)
        self.buff_size += 1
        if self.buff_size == self.K:
            for name in self.model.state_dict():
                if name not in self.named_parameters:
                    global_state[name] = torch.div(self.step[name], self.K)
                else:
                    global_state[name] += self.step[name]
            self.model.load_state_dict(global_state)
            self.global_step += 1
            self.buff_size = 0
            
        self.client_step[client_id] = self.global_step
        return global_state
    
    def compute_steps(self, client_id: Union[str, int], local_model: Union[Dict, OrderedDict],):
        """
        Compute changes to the global model after the aggregation.
        """
        if self.buff_size == 0:
            for name in self.model.state_dict():
                self.step[name] = torch.zeros_like(self.model.state_dict()[name])
        
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

        for name in self.model.state_dict():
            if name in self.named_parameters:
                self.step[name] += (
                    alpha_t * (-local_model[name]) if gradient_based
                    else alpha_t * (local_model[name] - self.model.state_dict()[name])
                )
            else:
                self.step[name] += local_model[name]
