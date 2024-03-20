import abc
from omegaconf import DictConfig
from concurrent.futures import Future
from typing import Union, Dict, Any, Tuple, OrderedDict

class BaseScheduler:
    def __init__(
        self, 
        scheduler_configs: DictConfig, 
        aggregator: Any,
        logger: Any
    ):
        self.scheduler_configs = scheduler_configs
        self.aggregator = aggregator
        self.logger = logger

    @abc.abstractmethod
    def schedule(self, client_id: Union[int, str], local_model: Union[Dict, OrderedDict], **kwargs) -> Union[Future, Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
        """
        Schedule the global aggregation for the local model from a client.
        :param local_model: the local model from a client
        :param client_idx: the index of the client
        :param kwargs: additional keyword arguments for the scheduler
        :return: the aggregated model or a future object for the aggregated model
        """
        pass

    @abc.abstractmethod
    def get_num_global_epochs(self) -> int:
        """Return the total number of global epochs for federated learning."""
        pass

    def get_parameters(self, **kwargs) -> Union[Future, Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
        """
        Return the global model to the clients. For the initial global model, the method can
        block until all clients have requested the initial global model to make sure all clients
        can get the same initial global model (if setting `same_init_model=True` in scheduler configs 
        and `kwargs['init_model']=True`).
        :params `kwargs['init_model']` (default is `True`): whether to get the initial global model or not
        :return the global model or a `Future` object for the global model
        """
        if (
            kwargs.get("init_model", True) 
            and self.scheduler_configs.get("same_init_model", True) 
            and (not kwargs.get("serial_run", False))
        ):
            if not hasattr(self, "init_model_requests"):
                self.init_model_requests = 0
                self.init_model_futures = []
            self.init_model_requests += 1
            future = Future()
            self.init_model_futures.append(future)
            if self.init_model_requests == self.scheduler_configs.num_clients:
                self.init_model_requests = 0
                init_model = self.aggregator.get_parameters(**kwargs)
                while self.init_model_futures:
                    future = self.init_model_futures.pop()
                    future.set_result(init_model)
            return future
        else:
            return self.aggregator.get_parameters(**kwargs)
