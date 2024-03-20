import threading
from omegaconf import DictConfig
from appfl.scheduler import BaseScheduler
from typing import Any, Union, Dict, OrderedDict, Tuple

class AsyncScheduler(BaseScheduler):
    def __init__(
        self, 
        scheduler_configs: DictConfig,
        aggregator: Any,
        logger: Any
    ):
        super().__init__(scheduler_configs, aggregator, logger)
        self._num_global_epochs = 0
        self._access_lock = threading.Lock()
    
    def schedule(self, client_id: Union[int, str], local_model: Union[Dict, OrderedDict], **kwargs) -> Union[Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
        """
        Schedule an asynchronous global aggregation for the local model from a client.
        The method will return the aggregated model immediately after the local model is submitted.
        :param local_model: the local model from a client
        :param client_id: the index of the client
        :param kwargs: additional keyword arguments for the scheduler
        :return: global_model: the aggregated model
        """
        with self._access_lock:
            global_model = self.aggregator.aggregate(client_id, local_model, **kwargs)
            self._num_global_epochs += 1
        return global_model
    
    def get_num_global_epochs(self) -> int:
        """Return the total number of global epochs for federated learning."""
        with self._access_lock:
            return self._num_global_epochs