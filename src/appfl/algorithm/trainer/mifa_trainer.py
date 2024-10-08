import copy
import torch
from torch.nn import Module
from omegaconf import DictConfig
from typing import Optional, Any
from torch.utils.data import Dataset
from appfl.algorithm.trainer import VanillaTrainer

class MIFATrainer(VanillaTrainer):
    """
    Trainer for the MIFA algorithm: https://proceedings.neurips.cc/paper/2021/file/64be20f6dd1dd46adf110cf871e3ed35-Paper.pdf
    """
    def __init__(
        self,
        model: Optional[Module]=None,
        loss_fn: Optional[Module]=None,
        metric: Optional[Any]=None,
        train_dataset: Optional[Dataset]=None,
        val_dataset: Optional[Dataset]=None,
        train_configs: DictConfig = DictConfig({}),
        logger: Optional[Any]=None,
        **kwargs
    ):
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            metric=metric,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_configs=train_configs,
            logger=logger,
            **kwargs
        )
        self.memory = copy.deepcopy(self.model)
        for name in self.model.state_dict():
            self.memory.state_dict()[name] = torch.zeros_like(self.model.state_dict()[name])
        
    def train(self):
        self.train_configs.send_gradient = True
        super().train()
        pseudo_grad = copy.deepcopy(self.model_state)
        for name in self.named_parameters:
            self.model_state[name] = self.pseudo_grad[name] - self.memory.state_dict()[name].cpu()
        self.memory.load_state_dict(pseudo_grad)