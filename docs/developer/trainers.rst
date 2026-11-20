Adding new trainers
===================

.. note::

    We always welcome your contributions to ``APPFL`` via pull requests.

To add new trainers to ``APPFL``, you can create your own trainer class by inheriting from ``appfl.algorithm.trainer.BaseTrainer`` and defining the following functions:

- ``train``: Perform local training on the local (sensitive) dataset.
- ``get_parameters``: Return the parameters to be sent to the server for aggregation.
- ``load_parameters``: [Optional] Load the aggregated parameters from the server.

.. note::

    - If your trainer does not need certain input parameters of ``BaseTrainer`` (e.g., you want to hardcode the loss function), simply leave them with their default value of ``None``.
    - If you need to add any other input parameters to the trainer, simply provide them in the ``**kwargs``.

.. code-block:: python

    class YourOwnTrainer(BaseTrainer):
        """
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
            model: Optional[nn.Module]=None,
            loss_fn: Optional[nn.Module]=None,
            metric: Optional[Any]=None,
            train_dataset: Optional[Dataset]=None,
            val_dataset: Optional[Dataset]=None,
            train_configs: DictConfig = DictConfig({}),
            logger: Optional[Any]=None,
            **kwargs
        ):
            self.round = 0
            self.model = model
            self.loss_fn = loss_fn
            self.metric = metric
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            self.train_configs = train_configs
            self.logger = logger
            self.__dict__.update(kwargs)

        def get_parameters(self) -> Union[Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
            """Return local model parameters and optional metadata."""
            pass

        def train(self):
            pass

        def load_parameters(self, params: Union[Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict], Any]):
            """Load model parameters. You can define your own way to load the parameters by overriding this function."""
            self.model.load_state_dict(params)

You may add any configuration parameters to your trainer and access them using ``self.train_configs.your_config_param``. When you start the FL experiment, you can specify the trainer configuration parameter values in the server configuration file in the following way:

.. code-block:: yaml

    client_configs:
        train_configs:
            trainer: "YourOwnTrainer"
            your_config_param_1: ...
            your_config_param_2: ...
            ...
