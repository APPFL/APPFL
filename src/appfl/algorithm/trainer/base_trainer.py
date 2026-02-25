import abc
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import Dataset
from typing import Optional, Dict, Any, Tuple, Union, OrderedDict


class BaseTrainer(abc.ABC):
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
        model: Optional[nn.Module] = None,
        loss_fn: Optional[nn.Module] = None,
        metric: Optional[Any] = None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        train_configs: Optional[DictConfig] = None,
        logger: Optional[Any] = None,
        client_id: Optional[Any] = None,
        **kwargs,
    ):
        if train_configs is None:
            train_configs = DictConfig({})
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
    ) -> Union[Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
        """Return local model parameters and optional metadata."""
        pass

    def load_parameters(
        self,
        params: Union[Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict], Any],
    ):
        """Load model parameters."""
        if isinstance(params, tuple):
            params = params[0]
        self.model.load_state_dict(params, strict=False)

    @abc.abstractmethod
    def train(self, **kwargs):
        pass

    def evaluate(self, **kwargs) -> Dict[str, Any]:
        """Evaluate the local model on a trainer-defined split/dataset."""
        del kwargs
        raise NotImplementedError("Trainer does not implement evaluate().")
