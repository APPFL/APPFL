import threading
from typing import Any, Union, Dict, OrderedDict
from concurrent.futures import Future
from omegaconf import DictConfig
from appfl.algorithm.scheduler import BaseScheduler
from appfl.algorithm.aggregator import BaseAggregator


class SyncScheduler(BaseScheduler):
    def __init__(
        self, scheduler_configs: DictConfig, aggregator: BaseAggregator, logger: Any
    ):
        super().__init__(scheduler_configs, aggregator, logger)
        self.local_models = {}
        self.aggregation_kwargs = {}
        self.future = {}
        self.num_clients = self.scheduler_configs.num_clients
        self._num_global_epochs = 0
        self._access_lock = threading.Lock()

    def schedule(
        self,
        client_id: Union[int, str],
        local_model: Union[Dict, OrderedDict],
        **kwargs,
    ) -> Future:
        """
        Schedule a synchronous global aggregation for the local model from a client.
        The method will return a future object for the aggregated model, which will
        be set after all clients have submitted their local models for the global aggregation.
        :param client_id: the id of the client
        :param local_model: the local model from a client
        :param kwargs: additional keyword arguments for the scheduler
        :return: the future object for the aggregated model
        """
        with self._access_lock:
            future = Future()
            self.local_models[client_id] = local_model
            for key, value in kwargs.items():
                if key not in self.aggregation_kwargs:
                    self.aggregation_kwargs[key] = {}
                self.aggregation_kwargs[key][client_id] = value
            self.future[client_id] = future
            if len(self.local_models) == self.num_clients:
                aggregated_model = self.aggregator.aggregate(
                    self.local_models, **self.aggregation_kwargs
                )
                while self.future:
                    client_id, future = self.future.popitem()
                    future.set_result(
                        self._parse_aggregated_model(aggregated_model, client_id)
                    )
                self.local_models.clear()
                self._num_global_epochs += 1
            return future

    def get_num_global_epochs(self) -> int:
        """
        Get the number of global epochs.
        :return: the number of global epochs
        """
        with self._access_lock:
            return self._num_global_epochs

    def _parse_aggregated_model(
        self, aggregated_model: Dict, client_id: Union[int, str]
    ) -> Dict:
        """
        Parse the aggregated model. Currently, this method is used to
        parse different client gradients for the vertical federated learning.
        :param aggregated_model: the aggregated model
        :return: the parsed aggregated model
        """
        if isinstance(aggregated_model, tuple):
            if client_id in aggregated_model[0]:
                return (
                    aggregated_model[0][client_id],
                    aggregated_model[1],
                )  # this is for vertical federated learning
            else:
                return aggregated_model
        else:
            if client_id in aggregated_model:
                return aggregated_model[
                    client_id
                ]  # this is for vertical federated learning
            else:
                return aggregated_model
