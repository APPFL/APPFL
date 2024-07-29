import io
import uuid
import torch
import threading
from appfl.scheduler import *
from appfl.aggregator import *
from concurrent.futures import Future
from omegaconf import DictConfig, OmegaConf
from appfl.logger import ServerAgentFileLogger
from typing import Dict, Optional, Union, OrderedDict, Tuple
from appfl.misc import create_instance_from_file, create_instance_from_file_source

class HFLNodeAgent:
    """
    `HFLNodeAgent` 
    This is the agent class for the HFL node which lies in between the HFL root server
    and leaf clients. It (partically) handles the requests from the leaf clients, and
    generates intermediate results to be sent to the server. It should be clearly noted
    that what the HFL node agent can achieve is limited compared to the server agent.
    For several tasks, to handle them, it requires the HFL node to use some type of
    connect communicator to send requests to the HFL root server, and loads responses 
    from the server.

    For example, in the case that the shared client configurations are needed, the HFL
    node agent is not supposed to have that information, and it must ask the server for
    it via certain communicator, and then call the `load_client_configs` method to get
    the information.
    """
    def __init__(
        self,
        hfl_node_agent_config: DictConfig = DictConfig({}),
    ) -> None:
        self.hfl_node_agent_config = hfl_node_agent_config
        self._create_logger()
        self._load_model()
        self._load_scheduler()
        self.closed_clients = set()
        self._init_model_available = False
        self._initial_model_lock = threading.Lock()
        self._client_configs_lock = threading.Lock()
        self._close_connection_lock = threading.Lock()
        
    def get_num_clients(self) -> int:
        """Return the number of clients."""
        self.num_clients = self.hfl_node_agent_config.scheduler_kwargs.num_clients
        return self.num_clients
        
    def get_id(self) -> str:
        """Return a unique node id for HFL root server to distinguish the node and clients."""
        if not hasattr(self, 'node_id'):
            self.node_id = str(uuid.uuid4())
        return self.node_id
        
    def load_client_configs(self, client_configs: DictConfig) -> None:
        """
        Load client configurations from the server.
        This method only functions once, and the shared client configurations will be stored.
        If updating the client configurations is needed, please invoke `update_client_configs`.
        
        :param config: Shared client configurations received from the server.
        """
        with self._client_configs_lock:
            if not hasattr(self.hfl_node_agent_config, "client_configs"):
                self.hfl_node_agent_config.client_configs = client_configs
                self._load_model()
                self._load_scheduler()

    def get_client_configs(self, **kwargs) -> Optional[DictConfig]:
        """
        Get client configurations that should be shared with all clients.
        
        :return: Shared lient configurations if available, otherwise `None`.
        """
        with self._client_configs_lock:
            if hasattr(self.hfl_node_agent_config, "client_configs"):
                return self.hfl_node_agent_config.client_configs
            else:
                return None
    
    def update_client_configs(self, client_configs: DictConfig) -> None:
        """
        Update client configurations from the server.
        :param config: Shared client configurations received from the server.
        """
        with self._client_configs_lock:
            self.hfl_node_agent_config.client_configs.update(client_configs)
            self._load_model()
            self._load_scheduler
    
    def load_init_parameters(self, model: Dict) -> None:
        """
        Load the initial global model parameters from the server.
        """
        if not self._init_model_available:
            with self._initial_model_lock:
                self.init_global_model = model
                self._init_model_available = True
                self.load_updated_model(model)
    
    def get_parameters(
        self,
        init_model: bool = False,
        **kwargs,
    ):
        """
        This function can only return the initial global model if  
        it is available. Otherwise, it will return `None`.
        """
        if not init_model:
            return None
        with self._initial_model_lock:
            return self.init_global_model if self._init_model_available else None

    def global_update(
        self,
        client_id: str,
        local_model: Union[Dict, OrderedDict, bytes],
        blocking: bool = False,
        **kwargs,
    ) -> Union[Future, Tuple[Union[Dict, OrderedDict], Dict], Union[Dict, OrderedDict]]:
        """
        Perform the intermediate aggregation and return the aggregated model.
        """
        if isinstance(local_model, bytes):
            local_model = self._bytes_to_model(local_model)
        aggregated_model = self.scheduler.schedule(client_id, local_model, **kwargs)
        metadata = {}
        if isinstance(aggregated_model, Future):
            if blocking:
                aggregated_model = aggregated_model.result()
            else:
                return aggregated_model
        if isinstance(aggregated_model, tuple):
            metadata = aggregated_model[1]
            aggregated_model = aggregated_model[0]
        return (aggregated_model, metadata) if len(metadata) > 0 else aggregated_model
        
    def load_updated_model(self, model: Union[Dict, OrderedDict]) -> None:
        """Load the updated model from the server."""
        self.aggregator.model.load_state_dict(model)
        
    def close_connection(self, client_id: Union[str, int]) -> None:
        """Close the connection with the client."""
        with self._close_connection_lock:
            self.closed_clients.add(client_id)
        
    def server_terminated(self) -> bool:
        with self._close_connection_lock:
            terminated = len(self.closed_clients) == self.num_clients
        return terminated
        
    def _load_model(self) -> None:
        """
        Load model from various sources with optional keyword arguments `model_kwargs`:
        - `model_path` and `model_name`: load model from a local file (usually for local simulation)
        - `model_source` and `model_name`: load model from a raw file source string (usually sent from the server)
        - Users can define their own way to load the model from other sources
        """
        if not hasattr(self.hfl_node_agent_config, "client_configs"):
            self.model = None
            return
        if not hasattr(self.hfl_node_agent_config.client_configs, "model_configs"):
            self.model = None
            return
        if hasattr(self.hfl_node_agent_config.client_configs.model_configs, "model_path") and hasattr(self.client_config.model_configs, "model_name"):
            kwargs = self.hfl_node_agent_config.client_configs.model_configs.get("model_kwargs", {})
            self.model = create_instance_from_file(
                self.hfl_node_agent_config.client_configs.model_configs.model_path,
                self.hfl_node_agent_config.client_configs.model_configs.model_name,
                **kwargs
            )
        elif hasattr(self.hfl_node_agent_config.client_configs.model_configs, "model_source") and hasattr(self.hfl_node_agent_config.client_configs.model_configs, "model_name"):
            kwargs = self.hfl_node_agent_config.client_configs.model_configs.get("model_kwargs", {})
            self.model = create_instance_from_file_source(
                self.hfl_node_agent_config.client_configs.model_configs.model_source,
                self.hfl_node_agent_config.client_configs.model_configs.model_name,
                **kwargs
            )
        else:
            self.model = None
            
    def _load_scheduler(self) -> None:
        """Load the scheduler and the aggregator."""
        if hasattr(self, "model") and self.model is not None:
            self.aggregator: BaseAggregator = eval(self.hfl_node_agent_config.aggregator)(
                self.model,
                OmegaConf.create(
                    self.hfl_node_agent_config.aggregator_kwargs if
                    hasattr(self.hfl_node_agent_config, "aggregator_kwargs") else {}
                ),
                self.logger,
            )
            self.scheduler: BaseScheduler = eval(self.hfl_node_agent_config.scheduler)(
                OmegaConf.create(
                    self.hfl_node_agent_config.scheduler_kwargs if 
                    hasattr(self.hfl_node_agent_config, "scheduler_kwargs") else {}
                ),
                self.aggregator,
                self.logger,
            )
        else:
            self.scheduler = None
            self.aggregator = None
            
    def _create_logger(self) -> None:
        """Create a logger for the HFL node agent."""
        kwargs = {}
        if hasattr(self.hfl_node_agent_config, "logging_output_dirname"):
            kwargs["file_dir"] = self.hfl_node_agent_config.logging_output_dirname
        if hasattr(self.hfl_node_agent_config, "logging_output_filename"):
            kwargs["file_name"] = self.hfl_node_agent_config.logging_output_filename
        self.logger = ServerAgentFileLogger(**kwargs)
        
    def _bytes_to_model(self, bytes: bytes) -> Dict:
        """Convert bytes to model."""
        return torch.load(io.BytesIO(bytes))