from typing import Any, Union, Dict, OrderedDict
from concurrent.futures import Future
from omegaconf import DictConfig
from appfl.scheduler import BaseScheduler

class SyncScheduler(BaseScheduler):
    def __init__(self, server_configs: DictConfig, aggregator: Any):
        super().__init__(server_configs, aggregator)
        self.local_models = {}
        self.future = {}
        assert (
            hasattr(self.server_config, "client_nums"), 
            f"{self.__class__.__name__}: client_nums attribute is not found in the server configuration."
        )
        self.client_nums = self.server_config.client_nums

    def schedule(self, local_model: Union[Dict, OrderedDict], client_idx: Union[int, str]) -> Future:
        """
        Schedule a synchronous global aggregation for the local model from a client.
        The method will return a future object for the aggregated model, which will
        be set after all clients have submitted their local models for the global aggregation.
        :param local_model: the local model from a client
        :param client_idx: the index of the client
        :return: the future object for the aggregated model
        """
        assert (
            client_idx not in self.local_models, 
            f"{self.__class__.__name__}: client {client_idx} has already submitted the local model."
        )
        future = Future()
        self.local_models[client_idx] = local_model
        self.future[client_idx] = future
        if len(self.local_models) == self.client_nums:
            aggregated_model = self.aggregator.aggregate(self.local_models)
            while self.future:
                client_idx, future = self.future.popitem()
                future.set_result(aggregated_model)
            self.local_models.clear()