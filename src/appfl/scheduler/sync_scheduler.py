from typing import Any, Union, Dict, OrderedDict
from concurrent.futures import Future
from omegaconf import DictConfig
from appfl.scheduler import BaseScheduler

class SyncScheduler(BaseScheduler):
    def __init__(
        self, 
        scheduler_configs: DictConfig, 
        aggregator: Any,
        logger: Any
    ):
        super().__init__(scheduler_configs, aggregator, logger)
        self.local_models = {}
        self.future = {}
        assert (
            hasattr(self.scheduler_configs, "num_clients"), 
            f"{self.__class__.__name__}: num_clients attribute is not found in the server configuration."
        )
        self.num_clients = self.scheduler_configs.num_clients
        self._num_global_epochs = 0

    def schedule(self, client_id: Union[int, str], local_model: Union[Dict, OrderedDict], **kwargs) -> Future:
        """
        Schedule a synchronous global aggregation for the local model from a client.
        The method will return a future object for the aggregated model, which will
        be set after all clients have submitted their local models for the global aggregation.
        :param client_id: the id of the client
        :param local_model: the local model from a client
        :param kwargs: additional keyword arguments for the scheduler
        :return: the future object for the aggregated model
        """
        assert (
            client_id not in self.local_models, 
            f"{self.__class__.__name__}: client {client_id} has already submitted the local model."
        )
        future = Future()
        self.local_models[client_id] = local_model
        self.future[client_id] = future
        if len(self.local_models) == self.num_clients:
            aggregated_model = self.aggregator.aggregate(self.local_models, **kwargs)
            while self.future:
                client_id, future = self.future.popitem()
                future.set_result(aggregated_model)
            self.local_models.clear()
            self._num_global_epochs += 1
        return future
    
    def get_num_global_epochs(self) -> int:
        """
        Get the number of global epochs.
        :return: the number of global epochs
        """
        return self._num_global_epochs