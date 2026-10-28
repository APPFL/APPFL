Adding new schedulers
=====================

.. note::

    We always welcome your contributions to ``APPFL`` via pull requests.

To add new schedulers to ``APPFL``, you can create your own scheduler class by inheriting from ``appfl.algorithm.scheduler.BaseScheduler`` and defining the following functions:

- ``schedule``: Take **one** ``client_id`` and **one** ``local_model`` from that client, and schedule a global aggregation of the client model with the aggregator. Return either the global model if the aggregation happens immediately, or a ``Future`` object otherwise.
- ``get_num_global_epochs``: Return the total number of global epochs (global updates) for the server to know when to stop the FL process.

.. code-block:: python

    class YourOwnScheduler(BaseScheduler):
        def __init__(
            self,
            scheduler_configs: DictConfig,
            aggregator: BaseAggregator,
            logger: Any
        ):
            self.scheduler_configs = scheduler_configs
            self.aggregator = aggregator
            self.logger = logger
            ...

        def schedule(self, client_id: Union[int, str], local_model: Union[Dict, OrderedDict], **kwargs) -> Union[Future, Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
            """
            Schedule the global aggregation for the local model from a client.
            :param client_id: the id of the client
            :param local_model: the local model from a client
            :param kwargs: additional keyword arguments for the scheduler
            :return: the aggregated model or a future object for the aggregated model
            """
            pass

        def get_num_global_epochs(self) -> int:
            """Return the total number of global epochs for federated learning."""
            pass

You may add any configuration parameters to your scheduler and access them using ``self.scheduler_configs.your_config_param``. When you start the FL experiment, you can specify the scheduler configuration parameter values in the server configuration file in the following way:

.. code-block:: yaml

    server_configs:
        ...
        scheduler: "YourOwnScheduler"
        scheduler_kwargs:
            num_clients: 2
            your_config_param_1: ...
            your_config_param_2: ...
        ...
