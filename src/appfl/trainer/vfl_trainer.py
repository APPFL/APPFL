import torch
import importlib
from typing import Any
from omegaconf import DictConfig
from torch.nn.modules import Module
from torch.utils.data import Dataset
from appfl.trainer import BaseTrainer

class VFLTrainer(BaseTrainer):
    def __init__(
        self, 
        model: Module | None = None, 
        train_dataset: Dataset | None = None, 
        val_dataset: Dataset | None = None, 
        train_configs: DictConfig = DictConfig({}),
        logger: Any | None = None, 
        **kwargs
    ):
        super().__init__(
            model=model, 
            train_dataset=train_dataset, 
            val_dataset=val_dataset, 
            train_configs=train_configs, 
            logger=logger, 
            **kwargs
        )
        
        if not hasattr(self.train_configs, "device"):
            self.train_configs.device = "cpu"
        
        self.model.to(self.train_configs.device)
            
        optim_module = importlib.import_module("torch.optim")
        assert hasattr(optim_module, self.train_configs.optim), \
            f"Optimizer {self.train_configs.optim} not found in torch.optim"
        self.optimizer = getattr(optim_module, self.train_configs.optim)(
            self.model.parameters(), 
            **self.train_configs.optim_args
        )
            
    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        self.train_embedding = self.model(self.train_dataset.to(self.train_configs.device))
        with torch.no_grad():
            self.model.eval()
            self.val_embedding = self.model(self.val_dataset.to(self.train_configs.device))
            
    def get_parameters(self):
        return {
            'train_embedding': self.train_embedding.detach().clone().cpu(),
            'val_embedding': self.val_embedding.cpu(),
        }
        
    def load_parameters(self, params):
        self.train_embedding.backward(params['client_grad'])
        self.optimizer.step()