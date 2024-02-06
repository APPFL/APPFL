import abc
import numpy as np
import torch.nn as nn
from omegaconf import DictConfig
from typing import Optional
from torch.utils.data import DataLoader
from appfl.misc import create_instance_from_file, get_function_from_file
from appfl.logger import ClientTrainerFileLogger

class BaseTrainer:
    """
    BaseTrainer:
        Abstract base trainer for FL clients.
    Args:
        model: torch neural network model to train
        train_dataloader: training data loader
        val_dataloader: validation data loader
        train_configs: training configurations
    """
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader]=None,
        train_configs: DictConfig = DictConfig({}),
    ):
        self.round = 0
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.train_configs = train_configs
        self.loss_fn = self._get_loss_fn()
        self.metric = self._get_metric()
        self._create_logger()

    def _create_logger(self):
        """
        Create logger for logging local training process
        You can overwrite this method to create your own logger.
        """
        kwargs = {}
        if hasattr(self.train_configs, "logging_id"):
            kwargs["logging_id"] = self.train_configs.logging_id
        if hasattr(self.train_configs, "output_dirname"):
            kwargs["file_dir"] = self.train_configs.output_dirname
        if hasattr(self.train_configs, "output_filename"):
            kwargs["file_name"] = self.train_configs.output_filename
        self.logger = ClientTrainerFileLogger(**kwargs)

    @abc.abstractmethod
    def get_parameters(self):
        """Return local model parameters"""
        pass

    @abc.abstractmethod
    def train(self):
        pass

    def _get_loss_fn(self):
        """Get loss function"""
        loss_fn = None
        if hasattr(self.train_configs, "loss_fn"):
            kwargs = self.train_configs.get("loss_fn_kwargs", {})
            if hasattr(nn, self.train_configs.loss_fn):                
                loss_fn = getattr(nn, self.train_configs.loss_fn)(**kwargs)
        elif hasattr(self.train_configs, "loss_fn_path") and hasattr(self.train_configs, "loss_fn_name"):
            loss_fn = create_instance_from_file(
                self.train_configs.loss_fn_path,
                self.train_configs.loss_fn_name,
                **self.train_configs.loss_fn_kwargs
            )
        if loss_fn is None:
            raise ValueError("Invalid loss function")
        return loss_fn

    def _get_metric(self):
        """Get evaluation metric."""
        metric = None
        if hasattr(self.train_configs, "metric_path") and hasattr(self.train_configs, "metric_name"):
            metric = get_function_from_file(
                self.train_configs.metric_path,
                self.train_configs.metric_name,
            )
        if metric is None:
            metric = self._default_metric
        return metric

    def _default_metric(self, y_true, y_pred):
        """Default metric: percentage of correct predictions"""
        if len(y_pred.shape) == 1:
            y_pred = np.round(y_pred)
        else:
            y_pred = y_pred.argmax(axis=1)
        return 100*np.sum(y_pred==y_true)/y_pred.shape[0]
