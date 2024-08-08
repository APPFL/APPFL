import os
import uuid
import pathlib
import importlib
import torch.nn as nn
from datetime import datetime
from proxystore.store import Store
from appfl.compressor import *
from appfl.trainer import BaseTrainer
from appfl.config import ClientAgentConfig
from omegaconf import DictConfig, OmegaConf
from typing import Union, Dict, OrderedDict, Tuple, Optional
from appfl.logger import ClientAgentFileLogger
from appfl.misc import create_instance_from_file, \
    run_function_from_file, \
    get_function_from_file, \
    create_instance_from_file_source, \
    get_function_from_file_source, \
    run_function_from_file_source

class ClientAgent:
    """
    The `ClientAgent` should act on behalf of the FL client to:
    - load configurations received from the server `ClientAgent.load_config`
    - get the size of the local dataset `ClientAgent.get_sample_size`
    - do the local training job using configurations `ClientAgent.train`
    - prepare data for communication `ClientAgent.get_parameters`
    - load parameters from the server `ClientAgent.load_parameters`
    - get a unique client id for server to distinguish clients `ClientAgent.get_id`

    Developers can add new methods to the client agent to support more functionalities,
    and use Fork + Pull Request to contribute to the project.

    Users can overwrite any class method to add custom functionalities of the client agent.
    """
    def __init__(
        self, 
        client_agent_config: ClientAgentConfig = ClientAgentConfig()
    ) -> None:
        self.client_agent_config = client_agent_config
        self._create_logger()
        self._load_model()
        self._load_loss()
        self._load_metric()
        self._load_data()
        self._load_trainer()
        self._load_compressor()
        self._load_proxystore()

    def load_config(self, config: DictConfig) -> None:
        """Load additional configurations provided by the server."""
        self.client_agent_config = OmegaConf.merge(self.client_agent_config, config)
        self._load_model()
        self._load_loss()
        self._load_metric()
        self._load_trainer()
        self._load_compressor()

    def get_id(self) -> str:
        """Return a unique client id for server to distinguish clients."""
        if not hasattr(self, 'client_id'):
            if hasattr(self.client_agent_config, "client_id"):
                self.client_id = self.client_agent_config.client_id
            else:
                self.client_id = str(uuid.uuid4())
        return self.client_id
    
    def get_sample_size(self) -> int:
        """Return the size of the local dataset."""
        return len(self.train_dataset)

    def train(self) -> None:
        """Train the model locally."""
        self.trainer.train()

    def get_parameters(self) -> Union[Dict, OrderedDict, bytes, Tuple[Union[Dict, OrderedDict, bytes], Dict]]:
        """Return parameters for communication"""
        params = self.trainer.get_parameters()
        params = {k: v.cpu() for k, v in params.items()}
        if isinstance(params, tuple):
            params, metadata = params
        else:
            metadata = None
        if self.enable_compression:
            params = self.compressor.compress_model(params)
        return self.proxy(params)[0] if metadata is None else (self.proxy(params)[0], metadata)
    
    def load_parameters(self, params) -> None:
        """Load parameters from the server."""
        self.trainer.load_parameters(params)
        
    def save_checkpoint(self, checkpoint_path: Optional[str]=None) -> None:
        """Save the model to a checkpoint file."""
        if checkpoint_path is None:
            output_dir = self.client_agent_config.train_configs.get("checkpoint_dirname", "./output")
            output_filename = self.client_agent_config.train_configs.get("checkpoint_filename", "checkpoint")
            curr_time_str = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
            checkpoint_path = f"{output_dir}/{output_filename}_{self.get_id()}_{curr_time_str}.pth"
            
        # Make sure the directory exists
        if not os.path.exists(os.path.dirname(checkpoint_path)):
            pathlib.Path(os.path.dirname(checkpoint_path)).mkdir(parents=True)
            
        torch.save(self.model.state_dict(), checkpoint_path)

    def proxy(self, obj):
        """
        Create the proxy of the object.
        :param obj: the object to be proxied.
        :return: the proxied object and a boolean value indicating whether the object is proxied.
        """
        if self.enable_proxystore:
            return self.proxystore.proxy(obj), True
        else:
            return obj, False
        
    def clean_up(self) -> None:
        """Clean up the client agent."""
        if hasattr(self, "proxystore") and self.proxystore is not None:
            try:
                self.proxystore.close(clear=True)
            except:
                self.proxystore.close()

    def _create_logger(self):
        """
        Create logger for the client agent to log local training process.
        You can modify or overwrite this method to create your own logger.
        """
        if hasattr(self, "logger"):
            return
        kwargs = {}
        if not hasattr(self.client_agent_config, "train_configs"):
            kwargs["logging_id"] = self.get_id()
            kwargs["file_dir"] = "./output"
            kwargs["file_name"] = "result"
        else:
            kwargs["logging_id"] = self.client_agent_config.train_configs.get("logging_id", self.get_id())
            kwargs["file_dir"] = self.client_agent_config.train_configs.get("logging_output_dirname", "./output")
            kwargs["file_name"] = self.client_agent_config.train_configs.get("logging_output_filename", "result")
        if hasattr(self.client_agent_config, "experiment_id"):
            kwargs["experiment_id"] = self.client_agent_config.experiment_id
        self.logger = ClientAgentFileLogger(**kwargs)

    def _load_data(self) -> None:
        """Get train and validation dataloaders from local dataloader file."""
        if hasattr(self.client_agent_config.data_configs, "dataset_source"):
            self.train_dataset, self.val_dataset = run_function_from_file_source(
                self.client_agent_config.data_configs.dataset_source,
                self.client_agent_config.data_configs.dataset_name,
                **(
                    self.client_agent_config.data_configs.dataset_kwargs 
                    if hasattr(self.client_agent_config.data_configs, "dataset_kwargs") 
                    else {}
                )
            )
        else:
            self.train_dataset, self.val_dataset = run_function_from_file(
                self.client_agent_config.data_configs.dataset_path,
                self.client_agent_config.data_configs.dataset_name,
                **(
                    self.client_agent_config.data_configs.dataset_kwargs
                    if hasattr(self.client_agent_config.data_configs, "dataset_kwargs")
                    else {}
                )
            )

    def _load_model(self) -> None:
        """
        Load model from various sources with optional keyword arguments `model_kwargs`:
        - `model_path` and `model_name`: load model from a local file (usually for local simulation)
        - `model_source` and `model_name`: load model from a raw file source string (usually sent from the server)
        - Users can define their own way to load the model from other sources
        """
        if not hasattr(self.client_agent_config, "model_configs"):
            self.model = None
            return
        if hasattr(self.client_agent_config.model_configs, "model_path") and hasattr(self.client_agent_config.model_configs, "model_name"):
            kwargs = self.client_agent_config.model_configs.get("model_kwargs", {})
            self.model = create_instance_from_file(
                self.client_agent_config.model_configs.model_path,
                self.client_agent_config.model_configs.model_name,
                **kwargs
            )
        elif hasattr(self.client_agent_config.model_configs, "model_source") and hasattr(self.client_agent_config.model_configs, "model_name"):
            kwargs = self.client_agent_config.model_configs.get("model_kwargs", {})
            self.model = create_instance_from_file_source(
                self.client_agent_config.model_configs.model_source,
                self.client_agent_config.model_configs.model_name,
                **kwargs
            )
        else:
            self.model = None

    def _load_loss(self) -> None:
        """
        Load loss function from various sources with optional keyword arguments `loss_fn_kwargs`:
        - `loss_fn_path` and `loss_fn_name`: load loss function from a local file (usually for local simulation)
        - `loss_fn_source` and `loss_fn_name`: load loss function from a raw file source string (usually sent from the server)
        - `loss_fn`: load commonly-used loss function from `torch.nn` module
        - Users can define their own way to load the loss function from other sources
        """
        if not hasattr(self.client_agent_config, "train_configs"):
            self.loss_fn = None
            return
        if hasattr(self.client_agent_config.train_configs, "loss_fn_path") and hasattr(self.client_agent_config.train_configs, "loss_fn_name"):
            kwargs = self.client_agent_config.train_configs.get("loss_fn_kwargs", {})
            self.loss_fn = create_instance_from_file(
                self.client_agent_config.train_configs.loss_fn_path,
                self.client_agent_config.train_configs.loss_fn_name,
                **kwargs
            )
        elif hasattr(self.client_agent_config.train_configs, "loss_fn"):
            kwargs = self.client_agent_config.train_configs.get("loss_fn_kwargs", {})
            if hasattr(nn, self.client_agent_config.train_configs.loss_fn):                
                self.loss_fn = getattr(nn, self.client_agent_config.train_configs.loss_fn)(**kwargs)
            else:
                self.loss_fn = None
        elif hasattr(self.client_agent_config.train_configs, "loss_fn_source") and hasattr(self.client_agent_config.train_configs, "loss_fn_name"):
            kwargs = self.client_agent_config.train_configs.get("loss_fn_kwargs", {})
            self.loss_fn = create_instance_from_file_source(
                self.client_agent_config.train_configs.loss_fn_source,
                self.client_agent_config.train_configs.loss_fn_name,
                **kwargs
            )
        else:
            self.loss_fn = None

    def _load_metric(self) -> None:
        """
        Load metric function from various sources:
        - `metric_path` and `metric_name`: load metric function from a local file (usually for local simulation)
        - `metric_source` and `metric_name`: load metric function from a raw file source string (usually sent from the server)
        - Users can define their own way to load the metric function from other sources
        """
        if not hasattr(self.client_agent_config, "train_configs"):
            self.metric = None
            return
        if hasattr(self.client_agent_config.train_configs, "metric_path") and hasattr(self.client_agent_config.train_configs, "metric_name"):
            self.metric = get_function_from_file(
                self.client_agent_config.train_configs.metric_path,
                self.client_agent_config.train_configs.metric_name
            )
        elif hasattr(self.client_agent_config.train_configs, "metric_source") and hasattr(self.client_agent_config.train_configs, "metric_name"):
            self.metric = get_function_from_file_source(
                self.client_agent_config.train_configs.metric_source,
                self.client_agent_config.train_configs.metric_name
            )
        else:
            self.metric = None

    def _load_trainer(self) -> None:
        """Obtain a local trainer"""
        if not hasattr(self.client_agent_config, "train_configs"):
            self.trainer = None
            return
        if not hasattr(self.client_agent_config.train_configs, "trainer"):
            self.trainer = None
            return
        trainer_module = importlib.import_module('appfl.trainer')
        if not hasattr(trainer_module, self.client_agent_config.train_configs.trainer):
            raise ValueError(f'Invalid trainer name: {self.client_agent_config.train_configs.trainer}')
        self.trainer: BaseTrainer = getattr(trainer_module, self.client_agent_config.train_configs.trainer)(
            model=self.model, 
            loss_fn=self.loss_fn,
            metric=self.metric,
            train_dataset=self.train_dataset, 
            val_dataset=self.val_dataset,
            train_configs=self.client_agent_config.train_configs,
            logger=self.logger,
        )

    def _load_compressor(self) -> None:
        """
        Create a compressor for compressing the model parameters.
        """
        if hasattr(self, "compressor") and self.compressor is not None:
            return
        self.compressor = None
        self.enable_compression = False
        if not hasattr(self.client_agent_config, "comm_configs"):
            return
        if not hasattr(self.client_agent_config.comm_configs, "compressor_configs"):
            return
        if getattr(self.client_agent_config.comm_configs.compressor_configs, "enable_compression", False):
            self.enable_compression = True
            self.compressor = eval(self.client_agent_config.comm_configs.compressor_configs.lossy_compressor)(
               self.client_agent_config.comm_configs.compressor_configs
            )

    def _load_proxystore(self) -> None:
        """
        Create the proxystore for storing and sending the model parameters from the client to the server.
        """
        if hasattr(self, "proxystore") and self.proxystore is not None:
            return
        self.proxystore = None
        self.enable_proxystore = False
        if not hasattr(self.client_agent_config, "comm_configs"):
            return
        if not hasattr(self.client_agent_config.comm_configs, "proxystore_configs"):
            return
        if getattr(self.client_agent_config.comm_configs.proxystore_configs, "enable_proxystore", False):
            self.enable_proxystore = True
            from proxystore.connectors.redis import RedisConnector
            from proxystore.connectors.file import FileConnector
            from proxystore.connectors.endpoint import EndpointConnector
            from appfl.communicator.connector.s3 import S3Connector
            self.proxystore = Store(
                self.get_id(),
                eval(self.client_agent_config.comm_configs.proxystore_configs.connector_type)(
                    **self.client_agent_config.comm_configs.proxystore_configs.connector_configs
                ),
            )