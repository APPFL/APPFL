import os
import copy
import torch
import pathlib
from datetime import datetime
from omegaconf import DictConfig
from appfl.aggregator import BaseAggregator
from typing import Union, Dict, OrderedDict, Any

class HFLFedAvgAggregator(BaseAggregator):
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
        self.round = 0
        self.lr = self.aggregator_config.get("lr", 1)
        self.logger.info(f"Initialized HFLFedAvgAggregator with lr={self.lr}")

    def get_parameters(self, **kwargs) -> Dict:
        return copy.deepcopy(self.model.state_dict())

    def aggregate(self, local_models: Dict[Union[str, int], Union[Dict, OrderedDict]], **kwargs) -> Dict:
        """
        Take the weighted average of local models from clients and return the global model.
        """
        self.round += 1
        global_state = copy.deepcopy(self.model.state_dict())
        
        num_clients_dict = kwargs.get("num_clients", {})
        if not hasattr(self, "total_num_clients"):
            self.total_num_clients = 0
            for client_id in local_models:
                self.total_num_clients += num_clients_dict.get(client_id, 1)
                self.logger.info(f"Client {client_id} has {num_clients_dict.get(client_id, 1)} samples.")
        
        self.compute_steps(local_models, num_clients_dict)
        
        for name in self.model.state_dict():
            if name not in self.named_parameters:
                param_sum = torch.zeros_like(self.model.state_dict()[name])
                for client_id, model in local_models.items():
                    param_sum += model[name] * num_clients_dict.get(client_id, 1)
                global_state[name] = torch.div(param_sum, self.total_num_clients)
            else:
                global_state[name] += self.step[name] * self.lr
            
        self.model.load_state_dict(global_state)
        
        # Save the global model if needed
        if self.aggregator_config.get("do_checkpoint", False):
            checkpoint_dir = self.aggregator_config.get('checkpoint_dirname', './output/checkpoints')
            checkpoint_filename = self.aggregator_config.get('checkpoint_filename', 'global_model')
            curr_time_str = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
            checkpoint_path = f"{checkpoint_dir}/{checkpoint_filename}_Server_{curr_time_str}.pth"
            # Create the directory if it does not exist
            if not os.path.exists(checkpoint_dir):
                pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            if self.round % self.aggregator_config.get("checkpoint_interval", 1) == 0:
                torch.save(self.model.state_dict(), checkpoint_path)
                self.logger.info(f"Saved global model to {checkpoint_path}")
        else:
            self.logger.info(f"Round {self.round} completed.")
        
        return global_state, {"num_clients": self.total_num_clients}
    
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
