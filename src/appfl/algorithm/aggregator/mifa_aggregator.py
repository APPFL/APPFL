import torch
from omegaconf import DictConfig
from appfl.algorithm.aggregator import FedAsyncAggregator
from typing import Union, Dict, OrderedDict, Any, Optional

class MIFAAggregator(FedAsyncAggregator):
    """
    MIFA Aggregator class for Federated Learning.
    For more details, check paper: https://proceedings.neurips.cc/paper/2021/file/64be20f6dd1dd46adf110cf871e3ed35-Paper.pdf
    """
    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        aggregator_configs: DictConfig = DictConfig({}),
        logger: Optional[Any] = None
    ):
        self.counter = 0
        self.model = model
        self.logger = logger
        self.aggregator_configs = aggregator_configs
        assert "K" in self.aggregator_configs, "K (buffer size) must be provided in the aggregator_configs"
        assert "num_clients" in self.aggregator_configs, "num_clients must be provided in the aggregator_configs"
        self.K = self.aggregator_configs.K
        self.num_clients = self.aggregator_configs.num_clients
        self.global_state = None # Models parameters that are used for aggregation, this is unknown at the beginning

    def aggregate(self, client_id: Union[str, int], local_model: Union[Dict, OrderedDict], **kwargs) -> Dict:
        if self.global_state is None:
            if self.model is not None:
                try: 
                    self.global_state = {
                        name: self.model.state_dict()[name] for name in local_model
                    }
                except:
                    self.global_state = {
                        name: tensor.detach().clone() for name, tensor in local_model.items()
                    }
            else:
                self.global_state = {
                    name: tensor.detach().clone() for name, tensor in local_model.items()
                }
            # Use pseudo_grad variable to store aggregator
            self.pseudo_grad = {
                name: torch.zeros_like(self.global_state[name]) for name in self.global_state
            }

        for name in self.global_state:
            self.pseudo_grad[name] += local_model[name]
        self.counter += 1
        if self.counter == self.K:
            for name in self.global_state:
                self.global_state[name] -= self.pseudo_grad[name] / self.num_clients
            self.counter = 0        
        return {k: v.clone() for k, v in self.global_state.items()}    
