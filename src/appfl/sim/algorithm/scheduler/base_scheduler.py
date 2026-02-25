import abc
from omegaconf import DictConfig
from concurrent.futures import Future
from typing import Union, Dict, Any, Tuple, OrderedDict
from appfl.sim.algorithm.aggregator import BaseAggregator


class BaseScheduler:
    def __init__(
        self, scheduler_configs: DictConfig, aggregator: BaseAggregator, logger: Any
    ):
        self.scheduler_configs = scheduler_configs
        self.aggregator = aggregator
        self.logger = logger

    @abc.abstractmethod
    def schedule(
        self,
        client_id: Union[int, str],
        local_model: Union[Dict, OrderedDict],
        **kwargs,
    ) -> Union[Future, Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
        """
        Schedule the global aggregation for the local model from a client.
        :param local_model: the local model from a client
        :param client_idx: the index of the client
        :param kwargs: additional keyword arguments for the scheduler
        :return: the aggregated model or a future object for the aggregated model
        """
        pass

    def get_parameters(
        self, **kwargs
    ) -> Union[Future, Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
        """
        Return the global model to the clients.
        """
        del kwargs
        return self.aggregator.get_parameters()
