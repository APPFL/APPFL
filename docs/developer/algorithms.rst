Adding new aggregators
======================

.. note::

    We always welcome your contributions to ``APPFL`` via pull requests.

To add new aggregators to ``APPFL``, you can create your own aggregator class by inheriting from ``appfl.algorithm.aggregator.BaseAggregator`` and defining the following functions:

- ``aggregate``: Take a list of local models (for synchronous FL), one local model (for asynchronous FL), or either (for semi-asynchronous FL) as input, and return the updated global model parameters.
- ``get_parameters``: Directly return the current global model parameters.

.. code-block:: python

    class YourOwnAggregator(BaseAggregator):
        def __init__(
            self,
            model: torch.nn.Module,
            aggregator_configs: DictConfig,
            logger: Any
        ):
            self.model = model
            self.aggregator_configs = aggregator_configs
            self.logger = logger
            ...

        def aggregate(self, *args, **kwargs) -> Union[Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
            """
            Aggregate local model(s) from clients and return the global model
            """
            pass

        def get_parameters(self, **kwargs) -> Union[Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
            """Return global model parameters"""
            pass

You may add any configuration parameters to your aggregator and access them using ``self.aggregator_configs.your_config_param``. When you start the FL experiment, you can specify the aggregator configuration parameter values in the server configuration file in the following way:

.. code-block:: yaml

    server_configs:
        ...
        aggregator: "YourOwnAggregator"
        aggregator_kwargs:
            your_config_param_1: ...
            your_config_param_2: ...
        ...
