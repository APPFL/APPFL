import uuid
import importlib
import torch.nn as nn
from omegaconf import DictConfig
from typing import Union, Dict, OrderedDict
from appfl.trainer import BaseTrainer
from appfl.compressor import Compressor
from appfl.config import ClientAgentConfig
from appfl.misc import create_instance_from_file, run_function_from_file

class APPFLClientAgent:
    """
    The `APPFLClientAgent` should act on behalf of the FL client to:
    - do the local training job using configurations `APPFLClientAgent.train`
    - prepare data for communication `APPFLClientAgent.get_parameters`
    - load parameters from the server `APPFLClientAgent.load_parameters`

    User can overwrite any class method to customize the behavior of the client agent.
    """
    def __init__(
        self, 
        client_agent_config: ClientAgentConfig = ClientAgentConfig()
    ) -> None:
        self.train_configs = client_agent_config.train_configs
        self.comm_configs = client_agent_config.comm_configs

        self._load_model(client_agent_config.model_configs)
        self._load_data(client_agent_config.data_configs)

        trainer_module = importlib.import_module('appfl.trainer')
        if not hasattr(trainer_module, self.train_configs.trainer):
            raise ValueError(f'Invalid trainer name: {self.train_configs.trainer}')
        self.trainer: BaseTrainer = getattr(trainer_module, self.train_configs.trainer)(
            self.model, 
            self.train_dataloader, 
            self.val_dataloader,
            self.train_configs,
        )

    def get_id(self) -> str:
        """Return a unique client id for server to distinguish clients."""
        if not hasattr(self, 'client_id'):
            self.client_id = str(uuid.uuid4())
        return self.client_id

    def train(self) -> None:
        """Train the model locally."""
        self.trainer.train()

    def get_parameters(self) -> Union[Dict, OrderedDict, bytes]:
        """Return parameters for communication"""
        params = self.trainer.get_parameters()
        if self.comm_configs.enable_compression:
            if not hasattr(self, 'compressor'):
                self.compressor = Compressor(self.comm_configs)
            params = self.compressor.compress_model(params)[0]
        return params
    
    def load_parameters(self, params) -> None:
        """Load parameters from the server."""
        self.model.load_state_dict(params)

    def _load_model(self, model_configs: DictConfig) -> None:
        """Load model from file."""
        self.model: nn.Module = create_instance_from_file(
            model_configs.model_path,
            model_configs.model_name,
            **model_configs.model_kwargs
        )

    def _load_data(self, data_configs: DictConfig) -> None:
        """Get train and validation dataloaders from local dataloader file."""
        self.train_dataloader, self.val_dataloader = run_function_from_file(
            data_configs.dataloader_path,
            data_configs.dataloader_name,
            **data_configs.dataloader_kwargs
        )

    