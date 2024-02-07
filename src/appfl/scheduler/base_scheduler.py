import abc
from omegaconf import DictConfig
from concurrent.futures import Future
from typing import Union, Dict, Any
from collections import OrderedDict

class BaseScheduler:
    def __init__(
        self, 
        server_config: DictConfig, 
        aggregator: Any,
        logger: Any
    ):
        self.server_config = server_config
        self.aggregator = aggregator
        self.logger = logger

    @abc.abstractmethod
    def schedule(self, client_id: Union[int, str], local_model: Union[Dict, OrderedDict]) -> Union[Future, Dict, OrderedDict]:
        """
        Schedule the global aggregation for the local model from a client.
        :param local_model: the local model from a client
        :param client_idx: the index of the client
        :return: the aggregated model or a future object for the aggregated model
        """
        pass
        
