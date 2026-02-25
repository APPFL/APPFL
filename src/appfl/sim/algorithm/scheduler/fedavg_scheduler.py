import gc
import threading
from typing import Any, Union, Dict, OrderedDict
from concurrent.futures import Future

from omegaconf import DictConfig

from appfl.sim.algorithm.scheduler.base_scheduler import BaseScheduler
from appfl.sim.algorithm.aggregator import BaseAggregator


class FedavgScheduler(BaseScheduler):
    def __init__(
        self, scheduler_configs: DictConfig, aggregator: BaseAggregator, logger: Any
    ):
        super().__init__(scheduler_configs, aggregator, logger)
        self.local_models = {}
        self.aggregation_kwargs = {}
        self.future = {}
        self.num_clients = self.scheduler_configs.num_clients
        self._access_lock = threading.Lock()

        self.optimize_memory = bool(scheduler_configs.get("optimize_memory", True))

    def schedule(
        self,
        client_id: Union[int, str],
        local_model: Union[Dict, OrderedDict],
        **kwargs,
    ) -> Future:
        with self._access_lock:
            future = Future()

            self.local_models[client_id] = local_model

            for key, value in kwargs.items():
                if key not in self.aggregation_kwargs:
                    self.aggregation_kwargs[key] = {}
                self.aggregation_kwargs[key][client_id] = value
            self.future[client_id] = future

            if len(self.local_models) == self.num_clients:
                if self.optimize_memory:
                    aggregated_model = self.aggregator.aggregate(
                        self.local_models, **self.aggregation_kwargs
                    )
                    temp_futures = dict(self.future)
                    self.local_models.clear()
                    self.aggregation_kwargs.clear()

                    while temp_futures:
                        cid, client_future = temp_futures.popitem()
                        client_future.set_result(
                            self._parse_aggregated_model(aggregated_model, cid)
                        )
                    self.future.clear()
                    gc.collect()
                else:
                    aggregated_model = self.aggregator.aggregate(
                        self.local_models, **self.aggregation_kwargs
                    )
                    while self.future:
                        cid, client_future = self.future.popitem()
                        client_future.set_result(
                            self._parse_aggregated_model(aggregated_model, cid)
                        )
                    self.local_models.clear()

            return future

    def _parse_aggregated_model(
        self, aggregated_model: Dict, client_id: Union[int, str]
    ) -> Dict:
        if isinstance(aggregated_model, tuple):
            if client_id in aggregated_model[0]:
                return (aggregated_model[0][client_id], aggregated_model[1])
            return aggregated_model
        if client_id in aggregated_model:
            return aggregated_model[client_id]
        return aggregated_model
