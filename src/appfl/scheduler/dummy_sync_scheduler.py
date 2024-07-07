import threading
from typing import Any, Union, Dict, OrderedDict, Optional
from concurrent.futures import Future
from omegaconf import DictConfig
from appfl.scheduler import BaseScheduler

class DummySyncScheduler(BaseScheduler):
    def __init__(
        self, 
        scheduler_configs: DictConfig, 
        aggregator: Any,
        logger: Any
    ):
        super().__init__(scheduler_configs, aggregator, logger)
        self.local_models = set()
        self.future = {}
        self.num_clients = self.scheduler_configs.num_clients
        self._num_global_epochs = 0
        self._access_lock = threading.Lock()

    def schedule(self, client_id: Union[int, str], local_model: Optional[Union[Dict, OrderedDict]], **kwargs) -> Future:
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
            if not hasattr(self, 'local_model'):
                self.local_model = self.aggregator.get_parameters()
            self.local_models.add(client_id)
            self.future[client_id] = future
            if len(self.local_models) == self.num_clients:
                while self.future:
                    client_id, future = self.future.popitem()
                    future.set_result(self.local_model)
                self.local_models = set()
                self._num_global_epochs += 1
                self.logger.info(f'The server receives models from all {self.num_clients} clients. Now it is epoch {self._num_global_epochs}')
            return future
    
    def get_num_global_epochs(self) -> int:
        """
        Get the number of global epochs.
        :return: the number of global epochs
        """
        with self._access_lock:
            return self._num_global_epochs