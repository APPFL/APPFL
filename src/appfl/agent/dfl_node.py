import uuid
import torch
import importlib
import threading
from concurrent.futures import Future
from typing import Dict, Union, OrderedDict, List
from omegaconf import DictConfig, OmegaConf
from appfl.algorithm.aggregator import *
from appfl.algorithm.trainer import BaseTrainer
from appfl.logger import ClientAgentFileLogger
from appfl.misc import (
    create_instance_from_file,
    run_function_from_file,
    get_function_from_file,
    create_instance_from_file_source,
    get_function_from_file_source,
    run_function_from_file_source,
)


class DFLNodeAgent:
    """
    `DFLNodeAgent`
    This is the agent class for the Decentralized FL (DFL) node. Unlike traditional federated learning,
    the DFL node acts both like a server and a client. It trains a local model locally and requires local
    models from neighbor nodes to update its model, and also servers for its neighbor nodes to send its
    own local model for their model updates.
    """

    def __init__(
        self,
        dfl_node_agent_config: DictConfig = DictConfig({}),
    ) -> None:
        self.dfl_node_agent_config = dfl_node_agent_config
        self._create_logger()
        self._load_model()
        self._load_loss()
        self._load_metric()
        self._load_data()
        self._load_trainer()
        self._load_aggregator()
        self._init_model_available = False
        self._closed_clients = set()
        self._current_model_params = None
        self._get_parameters_futures = []
        self._num_neighbors = self.dfl_node_agent_config.get("num_neighbors", None)
        self._model_params_lock = threading.Lock()
        self._close_connection_lock = threading.Lock()

    def get_id(self) -> str:
        """
        Return a unique node id other DFL nodes to distinguish the it.
        """
        if not hasattr(self, "node_id"):
            if hasattr(self.dfl_node_agent_config, "node_id"):
                self.node_id = self.dfl_node_agent_config.node_id
            else:
                self.node_id = str(uuid.uuid4())
        return self.node_id

    def train(self) -> None:
        """
        Train the local model.
        """
        self.trainer.train()
        # After training, update the model available flag and the current model parameters
        with self._model_params_lock:
            self._init_model_available = True
            self._current_model_params = self.trainer.get_parameters()
            for future in self._get_parameters_futures:
                future.set_result(self._current_model_params)
            self._get_parameters_futures = []

    def get_parameters(
        self,
        blocking: bool = True,
        **kwargs,
    ) -> Union[Dict, OrderedDict, Future]:
        """
        Get the local model parameters. This method will only return the local model parameters
        if it gets updated for at least one local round, otherwise it will either block until the
        local model is updated or return a future object for the local model parameters.
        """
        if self._init_model_available:
            with self._model_params_lock:
                return self._current_model_params
        else:
            future = Future()
            self._get_parameters_futures.append(future)
            return future.result() if blocking else future

    def aggregate_parameters(
        self, neighbor_models: Union[Dict[str, Dict], List[Dict]], **kwargs
    ) -> None:
        """
        Update trainer model parameters with the model parameters from neighboring nodes.
        :param neighbor_models: Model parameters from neighboring nodes.
        """
        if self._num_neighbors is None:
            self._num_neighbors = len(neighbor_models)
        new_model_params = self.aggregator.aggregate(
            self._current_model_params, neighbor_models, **kwargs
        )
        with self._model_params_lock:
            self.trainer.load_parameters(new_model_params)

    def close_connection(self, client_id: Union[str, int]) -> None:
        with self._close_connection_lock:
            self._closed_clients.add(client_id)

    def server_terminated(self) -> bool:
        if self._num_neighbors is None:
            return False
        with self._close_connection_lock:
            terminated = len(self._closed_clients) == self._num_neighbors
        return terminated

    def _create_logger(self) -> None:
        """Create a logger for the DFL node agent."""
        kwargs = {}
        if hasattr(self.dfl_node_agent_config, "logging_output_dirname"):
            kwargs["file_dir"] = self.dfl_node_agent_config.logging_output_dirname
        if hasattr(self.dfl_node_agent_config, "logging_output_filename"):
            kwargs["file_name"] = self.dfl_node_agent_config.logging_output_filename
        if hasattr(self.dfl_node_agent_config, "logging_id"):
            kwargs["logging_id"] = self.dfl_node_agent_config.logging_id
        self.logger = ClientAgentFileLogger(**kwargs)

    def _load_model(self) -> None:
        """
        Load model from various sources with optional keyword arguments `model_kwargs`:
        - `model_path` and `model_name`: load model from a local file (usually for local simulation)
        - `model_source` and `model_name`: load model from a raw file source string (usually sent from the server)
        - Users can define their own way to load the model from other sources
        """
        if not hasattr(self.dfl_node_agent_config, "model_configs"):
            self.model = None
            return
        self._set_seed()
        if hasattr(self.dfl_node_agent_config.model_configs, "model_path") and hasattr(
            self.dfl_node_agent_config.model_configs, "model_name"
        ):
            kwargs = self.dfl_node_agent_config.model_configs.get("model_kwargs", {})
            self.model = create_instance_from_file(
                self.dfl_node_agent_config.model_configs.model_path,
                self.dfl_node_agent_config.model_configs.model_name,
                **kwargs,
            )
        elif hasattr(
            self.dfl_node_agent_config.model_configs, "model_source"
        ) and hasattr(self.dfl_node_agent_config.model_configs, "model_name"):
            kwargs = self.dfl_node_agent_config.model_configs.get("model_kwargs", {})
            self.model = create_instance_from_file_source(
                self.dfl_node_agent_config.model_configs.model_source,
                self.dfl_node_agent_config.model_configs.model_name,
                **kwargs,
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
        if not hasattr(self.dfl_node_agent_config, "train_configs"):
            self.loss_fn = None
            return
        if hasattr(
            self.dfl_node_agent_config.train_configs, "loss_fn_path"
        ) and hasattr(self.dfl_node_agent_config.train_configs, "loss_fn_name"):
            kwargs = self.dfl_node_agent_config.train_configs.get("loss_fn_kwargs", {})
            self.loss_fn = create_instance_from_file(
                self.dfl_node_agent_config.train_configs.loss_fn_path,
                self.dfl_node_agent_config.train_configs.loss_fn_name,
                **kwargs,
            )
        elif hasattr(self.dfl_node_agent_config.train_configs, "loss_fn"):
            kwargs = self.dfl_node_agent_config.train_configs.get("loss_fn_kwargs", {})
            if hasattr(torch.nn, self.dfl_node_agent_config.train_configs.loss_fn):
                self.loss_fn = getattr(
                    torch.nn, self.dfl_node_agent_config.train_configs.loss_fn
                )(**kwargs)
            else:
                self.loss_fn = None
        elif hasattr(
            self.dfl_node_agent_config.train_configs, "loss_fn_source"
        ) and hasattr(self.dfl_node_agent_config.train_configs, "loss_fn_name"):
            kwargs = self.dfl_node_agent_config.train_configs.get("loss_fn_kwargs", {})
            self.loss_fn = create_instance_from_file_source(
                self.dfl_node_agent_config.train_configs.loss_fn_source,
                self.dfl_node_agent_config.train_configs.loss_fn_name,
                **kwargs,
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
        if not hasattr(self.dfl_node_agent_config, "train_configs"):
            self.metric = None
            return
        if hasattr(self.dfl_node_agent_config.train_configs, "metric_path") and hasattr(
            self.dfl_node_agent_config.train_configs, "metric_name"
        ):
            self.metric = get_function_from_file(
                self.dfl_node_agent_config.train_configs.metric_path,
                self.dfl_node_agent_config.train_configs.metric_name,
            )
        elif hasattr(
            self.dfl_node_agent_config.train_configs, "metric_source"
        ) and hasattr(self.dfl_node_agent_config.train_configs, "metric_name"):
            self.metric = get_function_from_file_source(
                self.dfl_node_agent_config.train_configs.metric_source,
                self.dfl_node_agent_config.train_configs.metric_name,
            )
        else:
            self.metric = None

    def _load_data(self) -> None:
        """Get train and validation dataloaders from local dataloader file."""
        if hasattr(self.dfl_node_agent_config.data_configs, "dataset_source"):
            self.train_dataset, self.val_dataset = run_function_from_file_source(
                self.dfl_node_agent_config.data_configs.dataset_source,
                self.dfl_node_agent_config.data_configs.dataset_name,
                **(
                    self.dfl_node_agent_config.data_configs.dataset_kwargs
                    if hasattr(
                        self.dfl_node_agent_config.data_configs, "dataset_kwargs"
                    )
                    else {}
                ),
            )
        else:
            self.train_dataset, self.val_dataset = run_function_from_file(
                self.dfl_node_agent_config.data_configs.dataset_path,
                self.dfl_node_agent_config.data_configs.dataset_name,
                **(
                    self.dfl_node_agent_config.data_configs.dataset_kwargs
                    if hasattr(
                        self.dfl_node_agent_config.data_configs, "dataset_kwargs"
                    )
                    else {}
                ),
            )

    def _load_trainer(self) -> None:
        """Obtain a local trainer"""
        if not hasattr(self.dfl_node_agent_config, "train_configs"):
            self.trainer = None
            return
        if not hasattr(self.dfl_node_agent_config.train_configs, "trainer"):
            self.trainer = None
            return
        trainer_module = importlib.import_module("appfl.algorithm.trainer")
        if not hasattr(
            trainer_module, self.dfl_node_agent_config.train_configs.trainer
        ):
            raise ValueError(
                f"Invalid trainer name: {self.dfl_node_agent_config.train_configs.trainer}"
            )
        self.trainer: BaseTrainer = getattr(
            trainer_module, self.dfl_node_agent_config.train_configs.trainer
        )(
            model=self.model,
            loss_fn=self.loss_fn,
            metric=self.metric,
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            train_configs=self.dfl_node_agent_config.train_configs,
            logger=self.logger,
        )

    def _load_aggregator(self) -> None:
        self.aggregator: BaseAggregator = eval(self.dfl_node_agent_config.aggregator)(
            self.model,
            OmegaConf.create(
                self.dfl_node_agent_config.aggregator_kwargs
                if hasattr(self.dfl_node_agent_config, "aggregator_kwargs")
                else {}
            ),
            self.logger,
        )

    def _set_seed(self):
        """
        This function makes sure that all clients have the same initial model parameters.
        """
        seed_value = self.dfl_node_agent_config.model_configs.get("seed", 42)
        torch.manual_seed(seed_value)  # Set PyTorch seed
        torch.cuda.manual_seed_all(seed_value)  # Set seed for all GPUs
        torch.backends.cudnn.deterministic = True  # Use deterministic algorithms
        torch.backends.cudnn.benchmark = False  # Disable this to ensure reproducibility
