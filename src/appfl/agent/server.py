from appfl.scheduler import *
from appfl.aggregator import *
from concurrent.futures import Future
from appfl.config import ServerAgentConfig
from omegaconf import OmegaConf, DictConfig
from typing import Union, Dict, OrderedDict
from appfl.logger import ServerAgentFileLogger
from appfl.misc import create_instance_from_file

class APPFLServerAgent:
    """
    `APPFLServerAgent` should act on behalf of the FL server to:
    - provide configurations that are shared among all clients to the clients (e.g. trainer, model, etc.) `APPFLServerAgent.get_client_configs`
    - take the local model from a client, update the global model, and return it `APPFLServerAgent.global_update`
    - provide the global model to the clients (no input and no aggregation) `APPFLServerAgent.get_parameters`

    User can overwrite any class method to customize the behavior of the client agent.
    """
    def __init__(
        self,
        server_agent_config: ServerAgentConfig = ServerAgentConfig()
    ) -> None:
        self.server_agent_config = server_agent_config
        assert (
            hasattr(self.server_agent_config.server_configs, "aggregator"), 
            f"{self.__class__.__name__}: aggregator attribute is not found in the server agent configuration."
        )
        assert (
            hasattr(self.server_agent_config.server_configs, "scheduler"), 
            f"{self.__class__.__name__}: scheduler attribute is not found in the server agent configuration."
        )
        assert (
            hasattr(self.server_agent_config.client_configs, "model_configs"), 
            f"{self.__class__.__name__}: model_configs attribute is not found in the client agent configuration."
        )
        self._create_logger()
        self._load_model(self.server_agent_config.client_configs.model_configs)
        self.aggregator: BaseAggregator = self._get_aggregator()
        self.scheduler: BaseScheduler = self._get_scheduler()

    def get_client_configs(self) -> DictConfig:
        """Return the FL configurations that are shared among all clients."""
        return self.server_agent_config.client_configs
    
    def global_update(
            self, 
            client_id: Union[int, str],
            local_model: Union[Dict, OrderedDict],
            blocking: bool = False
        ) -> Union[Dict, OrderedDict, Future]:
        """
        Update the global model using the local model from a client and return the updated global model.
        :param: client_id: A unique client id for server to distinguish clients, which be obtained via `ClientAgent.get_id()`.
        :param: local_model: The local model from a client.
        :param: blocking: The global model may not be immediately available for certain aggregation methods (e.g. any synchronous method).
            Setting `blocking` to `True` will block the client until the global model is available. 
            Otherwise, the method may return a `Future` object if the most up-to-date global model is not yet available.
        """
        global_model = self.scheduler.schedule(client_id, local_model)
        if not isinstance(global_model, Future):
            return global_model
        if blocking:
            return global_model.result() # blocking until the `Future` is done
        else:
            return global_model # return the `Future` object
        
    def get_parameters(self) -> Union[Dict, OrderedDict]:
        """Return the global model to the clients."""
        return self.aggregator.get_parameters()

    def _create_logger(self) -> None:
        kwargs = {}
        if hasattr(self.server_agent_config.server_configs, "logging_output_dirname"):
            kwargs["file_dir"] = self.server_agent_config.server_configs.logging_output_dirname
        if hasattr(self.server_agent_config.server_configs, "logging_output_filename"):
            kwargs["file_name"] = self.server_agent_config.server_configs.logging_output_filename
        self.logger = ServerAgentFileLogger(**kwargs)

    def _load_model(self, model_configs: DictConfig) -> None:
        """Load model from the definition file."""
        self.model = create_instance_from_file(
            model_configs.model_path,
            model_configs.model_name,
            **model_configs.model_kwargs
        )

    def _get_aggregator(self) -> BaseAggregator:
        """Obtain the global aggregator."""
        return eval(self.server_agent_config.server_configs.aggregator)(
            self.model,
            OmegaConf.create(self.server_agent_config.server_configs.aggregator_args),
            self.logger,
        )
        
    def _get_scheduler(self) -> BaseScheduler:
        """Obtain the scheduler."""
        return eval(self.server_agent_config.server_configs.scheduler)(
            OmegaConf.create(self.server_agent_config.server_configs.scheduler_args),
            self.aggregator,
            self.logger,
        )
