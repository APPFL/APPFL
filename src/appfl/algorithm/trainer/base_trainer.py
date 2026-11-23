import abc
from collections import OrderedDict
from typing import Any

from omegaconf import DictConfig
from torch import nn
from torch.utils.data import Dataset


class BaseTrainer:
    """
    BaseTrainer:
        Abstract base trainer for FL clients.
    Args:
        model: torch neural network model to train
        loss_fn: loss function for the model training
        metric: metric function for the model evaluation
        train_dataset: training dataset
        val_dataset: validation dataset
        train_configs: training configurations
        logger: logger for the trainer
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        loss_fn: nn.Module | None = None,
        metric: Any | None = None,
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        train_configs: DictConfig = DictConfig({}),
        logger: Any | None = None,
        client_id: Any | None = None,
        **kwargs,
    ):
        self.round = 0
        self.model = model
        self.loss_fn = loss_fn
        self.metric = metric
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_configs = train_configs
        self.logger = logger
        self.client_id = client_id
        self.__dict__.update(kwargs)

    @abc.abstractmethod
    def get_parameters(
        self,
    ) -> dict | OrderedDict | tuple[dict | OrderedDict, dict]:
        """Return local model parameters and optional metadata."""

    @abc.abstractmethod
    def train(self, **kwargs):
        pass

    def load_parameters(
        self,
        params: dict | OrderedDict | tuple[dict | OrderedDict, dict] | Any,
    ):
        """Load model parameters."""
        self.model.load_state_dict(params, strict=False)
