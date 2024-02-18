import torch.nn as nn
from appfl.scheduler import *
from appfl.aggregator import *
from concurrent.futures import Future
from appfl.config import ServerAgentConfig
from omegaconf import OmegaConf, DictConfig
from appfl.logger import ServerAgentFileLogger
from typing import Union, Dict, OrderedDict, Tuple
from appfl.misc import create_instance_from_file, get_function_from_file

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
        self._create_logger()
        self._load_model()
        self._load_loss()
        self._load_metric()
        self._get_scheduler()

    def get_client_configs(self, **kwargs) -> DictConfig:
        """Return the FL configurations that are shared among all clients."""
        return self.server_agent_config.client_configs
    
    def global_update(
            self, 
            client_id: Union[int, str],
            local_model: Union[Dict, OrderedDict],
            blocking: bool = False,
            **kwargs
        ) -> Union[Future, Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
        """
        Update the global model using the local model from a client and return the updated global model.
        :param: client_id: A unique client id for server to distinguish clients, which be obtained via `ClientAgent.get_id()`.
        :param: local_model: The local model from a client.
        :param: blocking: The global model may not be immediately available for certain aggregation methods (e.g. any synchronous method).
            Setting `blocking` to `True` will block the client until the global model is available. 
            Otherwise, the method may return a `Future` object if the most up-to-date global model is not yet available.
        """
        global_model = self.scheduler.schedule(client_id, local_model, **kwargs)
        if not isinstance(global_model, Future):
            return global_model
        if blocking:
            return global_model.result() # blocking until the `Future` is done
        else:
            return global_model # return the `Future` object
        
    def get_parameters(self, **kwargs) -> Union[Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
        """Return the global model to the clients."""
        global_model = self.scheduler.get_parameters(**kwargs)
        if not isinstance(global_model, Future):
            return global_model
        else:
            return global_model.result()

    def _create_logger(self) -> None:
        kwargs = {}
        if hasattr(self.server_agent_config.server_configs, "logging_output_dirname"):
            kwargs["file_dir"] = self.server_agent_config.server_configs.logging_output_dirname
        if hasattr(self.server_agent_config.server_configs, "logging_output_filename"):
            kwargs["file_name"] = self.server_agent_config.server_configs.logging_output_filename
        self.logger = ServerAgentFileLogger(**kwargs)

    def _load_model(self) -> None:
        """
        Load model from the definition file, and read the source code of the model for sendind to the client.
        User can overwrite this method to load the model from other sources.
        """
        model_configs = self.server_agent_config.client_configs.model_configs
        self.model = create_instance_from_file(
            model_configs.model_path,
            model_configs.model_name,
            **model_configs.model_kwargs
        )
        # load the model source file and delete model path
        with open(model_configs.model_path, 'r') as f:
            self.server_agent_config.client_configs.model_configs.model_source = f.read()
        del self.server_agent_config.client_configs.model_configs.model_path

    def _load_loss(self) -> None:
        """
        Load loss function from various sources.
        - `loss_fn_path` and `loss_fn_name`: load the loss function from a file.
        - `loss_fn`: load the loss function from `torch.nn` module.
        - Users can define their own way to load the loss function from other sources.
        """
        if hasattr(self.server_agent_config.client_configs.train_configs, "loss_fn_path") and hasattr(self.server_agent_config.client_configs.train_configs, "loss_fn_name"):
            kwargs = self.server_agent_config.client_configs.train_configs.get("loss_fn_kwargs", {})
            self.loss_fn = create_instance_from_file(
                self.server_agent_config.client_configs.train_configs.loss_fn_path,
                self.server_agent_config.client_configs.train_configs.loss_fn_name,
                **kwargs
            )
            with open(self.server_agent_config.client_configs.train_configs.loss_fn_path, 'r') as f:
                self.server_agent_config.client_configs.train_configs.loss_fn_source = f.read()
            del self.server_agent_config.client_configs.train_configs.loss_fn_path
        elif hasattr(self.server_agent_config.client_configs.train_configs, "loss_fn"):
            kwargs = self.server_agent_config.client_configs.train_configs.get("loss_fn_kwargs", {})
            if hasattr(nn, self.server_agent_config.client_configs.train_configs.loss_fn):
                self.loss_fn = getattr(nn, self.server_agent_config.client_configs.train_configs.loss_fn)(**kwargs)
            else:
                self.loss_fn = None
        else:
            self.loss_fn = None

    def _load_metric(self) -> None:
        """
        Load metric function from a file.
        User can define their own way to load the metric function from other sources.
        """
        if hasattr(self.server_agent_config.client_configs.train_configs, "metric_path") and hasattr(self.server_agent_config.client_configs.train_configs, "metric_name"):
            self.metric = get_function_from_file(
                self.server_agent_config.client_configs.train_configs.metric_path,
                self.server_agent_config.client_configs.train_configs.metric_name,
            )
            with open(self.server_agent_config.client_configs.train_configs.metric_path, 'r') as f:
                self.server_agent_config.client_configs.train_configs.metric_source = f.read()
            del self.server_agent_config.client_configs.train_configs.metric_path
        else:
            self.metric = None

    def _get_scheduler(self) -> None:
        """Obtain the scheduler."""
        self._aggregator: BaseAggregator = eval(self.server_agent_config.server_configs.aggregator)(
            self.model,
            OmegaConf.create(self.server_agent_config.server_configs.aggregator_kwargs),
            self.logger,
        )
        self.scheduler: BaseScheduler = eval(self.server_agent_config.server_configs.scheduler)(
            OmegaConf.create(self.server_agent_config.server_configs.scheduler_kwargs),
            self._aggregator,
            self.logger,
        )
