import abc
import torch.nn as nn
from omegaconf import DictConfig
from typing import Optional, Dict, Any
from torch.utils.data import DataLoader

class BaseTrainer:
    """
    BaseTrainer:
        Abstract base trainer for FL clients.
    Args:
        model: torch neural network model to train
        loss_fn: loss function for the model training
        metric: metric function for the model evaluation
        train_dataloader: training data loader
        val_dataloader: validation data loader
        train_configs: training configurations
        logger: logger for the trainer
    """
    def __init__(
        self,
        model: Optional[nn.Module]=None,
        loss_fn: Optional[nn.Module]=None,
        metric: Optional[Any]=None,
        train_dataloader: Optional[DataLoader]=None,
        val_dataloader: Optional[DataLoader]=None,
        train_configs: DictConfig = DictConfig({}),
        logger: Optional[Any]=None,
        **kwargs
    ):
        self.round = 0
        self.model = model
        self.loss_fn = loss_fn
        self.metric = metric
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.train_configs = train_configs
        self.logger = logger
        self.__dict__.update(kwargs)

    @abc.abstractmethod
    def get_parameters(self) -> Dict:
        """Return local model parameters"""
        pass

    @abc.abstractmethod
    def train(self):
        pass
