import importlib
from omegaconf import DictConfig
from appfl.trainer import BaseTrainer
from appfl.compressor import Compressor
from appfl.config import ClientAgentConfig
from appfl.misc import create_instance_from_file, run_function_from_file

class APPFLClientAgent:
    def __init__(self, client_agent_config: ClientAgentConfig = ClientAgentConfig()):
        self.train_configs = client_agent_config.train_configs
        self.comm_configs = client_agent_config.comm_configs

        self.load_model(client_agent_config.model_configs)
        self.load_data(client_agent_config.data_configs)

        trainer_module = importlib.import_module('appfl.trainer')
        if not hasattr(trainer_module, self.train_configs.trainer):
            raise ValueError(f'Invalid trainer name: {self.train_configs.trainer}')
        self.trainer: BaseTrainer = getattr(trainer_module, self.train_configs.trainer)(
            self.model, 
            self.train_dataloader, 
            self.val_dataloader,
            self.train_configs,
        )

    def load_model(self, model_configs: DictConfig):
        self.model = create_instance_from_file(
            model_configs.model_path,
            model_configs.model_name,
            **model_configs.model_kwargs
        )

    def load_data(self, data_configs: DictConfig):
        self.train_dataloader, self.val_dataloader = run_function_from_file(
            data_configs.dataloader_path,
            data_configs.dataloader_name,
            **data_configs.dataloader_kwargs
        )

    def train(self):
        self.trainer.train()

    def get_parameters(self):
        params = self.trainer.get_parameters()
        if self.comm_configs.enable_compression:
            compressor = Compressor(self.comm_configs)
            params = compressor.compress_model(params)[0]
        return params