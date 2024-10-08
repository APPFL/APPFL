import io
import torch
import pathlib
import threading
import torch.nn as nn
from appfl.compressor import *
from appfl.algorithm.scheduler import *
from appfl.algorithm.aggregator import *
from appfl.config import ServerAgentConfig
from appfl.logger import ServerAgentFileLogger
from appfl.misc import create_instance_from_file, get_function_from_file, run_function_from_file
from concurrent.futures import Future
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig
from typing import Union, Dict, OrderedDict, Tuple, Optional
from appfl.misc.data_readiness.report import *

class ServerAgent:
    """
    `ServerAgent` should act on behalf of the FL server to:
    - provide configurations that are shared among all clients to the clients (e.g. trainer, model, etc.) `ServerAgent.get_client_configs`
    - take the local model from a client, update the global model, and return it `ServerAgent.global_update`
    - provide the global model to the clients (no input and no aggregation) `ServerAgent.get_parameters`

    User can overwrite any class method to customize the behavior of the server agent.
    """
    def __init__(
        self,
        server_agent_config: ServerAgentConfig = ServerAgentConfig()
    ) -> None:
        self.server_agent_config = server_agent_config
        if hasattr(self.server_agent_config.client_configs, "comm_configs"):
            self.server_agent_config.server_configs.comm_configs = (OmegaConf.merge(
                self.server_agent_config.server_configs.comm_configs,
                self.server_agent_config.client_configs.comm_configs
            ) if hasattr(self.server_agent_config.server_configs, "comm_configs") 
            else self.server_agent_config.client_configs.comm_configs
            )
        self._create_logger()
        self._load_model()
        self._load_loss()
        self._load_metric()
        self._load_trainer()
        self._load_scheduler()
        self._load_compressor()
        self._load_val_data()
        
    def get_num_clients(self) -> int:
        """
        Get the number of clients.
        """
        if not hasattr(self, "num_clients"):
            assert (
                hasattr(self.server_agent_config.server_configs, "num_clients") or
                hasattr(self.server_agent_config.server_configs.scheduler_kwargs, "num_clients") or
                hasattr(self.server_agent_config.server_configs.aggregator_kwargs, "num_clients")
            ), "The number of clients should be set in the server configurations."
            self.num_clients = (
                self.server_agent_config.server_configs.num_clients if
                hasattr(self.server_agent_config.server_configs, "num_clients") else
                self.server_agent_config.server_configs.scheduler_kwargs.num_clients if
                hasattr(self.server_agent_config.server_configs.scheduler_kwargs, "num_clients") else
                self.server_agent_config.server_configs.aggregator_kwargs.num_clients
            )
        return self.num_clients

    def get_client_configs(self, **kwargs) -> DictConfig:
        """Return the FL configurations that are shared among all clients."""
        return self.server_agent_config.client_configs

    def global_update(
        self, 
        client_id: Union[int, str],
        local_model: Union[Dict, OrderedDict, bytes],
        blocking: bool = False,
        **kwargs
    ) -> Union[Future, Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
        """
        Update the global model using the local model from a client and return the updated global model.
        :param: client_id: A unique client id for server to distinguish clients, which be obtained via `ClientAgent.get_id()`.
        :param: local_model: The local model from a client, can be serailzed bytes.
        :param: blocking: The global model may not be immediately available for certain aggregation methods (e.g. any synchronous method).
            Setting `blocking` to `True` will block the client until the global model is available. 
            Otherwise, the method may return a `Future` object if the most up-to-date global model is not yet available.
        :return: The updated global model (as a Dict or OrderedDict), and optional metadata (as a Dict) if `blocking` is `True`.
            Otherwise, return the `Future` object of the updated global model and optional metadata.
        """
        if self.training_finished():
            global_model = self.scheduler.get_parameters(init_model=False)
            return global_model
        else:
            if isinstance(local_model, bytes):
                local_model = self._bytes_to_model(local_model)
            global_model = self.scheduler.schedule(client_id, local_model, **kwargs)
            if not isinstance(global_model, Future):
                return global_model
            if blocking:
                return global_model.result() # blocking until the `Future` is done
            else:
                return global_model # return the `Future` object
        
    def get_parameters(
        self, 
        blocking: bool = False,
        **kwargs
    ) -> Union[Future, Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
        """
        Return the global model to the clients.
        :param: `blocking`: The global model may not be immediately available (e.g. if the server wants to wait for all client
            to send the `get_parameters` request before returning the global model for same model initialization). 
            Setting `blocking` to `True` will block the client until the global model is available. 
        :param: `kwargs`: Additional arguments for the method. Specifically,
            - `init_model`: whether getting the initial model (which should be same among all clients, thus blocking)
            - `serial_run`: set `True` if for serial simulation run, thus no blocking is needed.
            - `globus_compute_run`: set `True` if for globus compute run, thus no blocking is needed.
        """
        global_model = self.scheduler.get_parameters(**kwargs)
        if not isinstance(global_model, Future):
            return global_model
        if blocking:
            return global_model.result() # blocking until the `Future` is done
        else:
            return global_model # return the `Future` object
        
    def set_sample_size(
            self, 
            client_id: Union[int, str],
            sample_size: int,
            sync: bool = False,
            blocking: bool = False,
        ) -> Optional[Union[Dict, Future]]:
        """
        Set the size of the local dataset of a client.
        :param: client_id: A unique client id for server to distinguish clients, which can be obtained via `ClientAgent.get_id()`.
        :param: sample_size: The size of the local dataset of a client.
        :param: sync: Whether to synchronize the sample size among all clients. If `True`, the method can return the relative weight of the client.
        :param: blocking: Whether to block the client until the sample size of all clients is synchronized. 
            If `True`, the method will return the relative weight of the client.
            Otherwise, the method may return a `Future` object of the relative weight, which will be resolved 
            when the sample size of all clients is synchronized.
        """
        self.aggregator.set_client_sample_size(client_id, sample_size)
        if sync:
            if not hasattr(self, "_client_sample_size"):
                self._client_sample_size = {}
                self._client_sample_size_future = {}
                self._client_sample_size_lock = threading.Lock()
            with self._client_sample_size_lock:
                self._client_sample_size[client_id] = sample_size
                future = Future()
                self._client_sample_size_future[client_id] = future
                if len(self._client_sample_size) == self.get_num_clients():
                    total_sample_size = sum(self._client_sample_size.values())
                    for client_id in self._client_sample_size_future:
                        self._client_sample_size_future[client_id].set_result(
                            {"client_weight": self._client_sample_size[client_id] / total_sample_size}
                        )
                    self._client_sample_size = {}
                    self._client_sample_size_future = {}
            if blocking:
                return future.result()
            else:
                return future
        return None

    def server_validate(self):
        """
        Validate the server model using the validation dataset.
        """
        if not hasattr(self, "_val_dataset"):
            self.logger.info("No validation dataset is provided.")
            return None
        else:
            return self._validate()

    def training_finished(self, **kwargs) -> bool:
        """Indicate whether the training is finished."""
        return self.server_agent_config.server_configs.num_global_epochs <= self.scheduler.get_num_global_epochs()
    
    def close_connection(self, client_id: Union[int, str]) -> None:
        """Record the client that has finished the communication with the server."""
        if not hasattr(self, 'closed_clients'):
            self.closed_clients = set()
            self._close_connection_lock = threading.Lock()
        with self._close_connection_lock:
            self.closed_clients.add(client_id)

    def data_readiness_report(self, readiness_report: Dict) -> None:
        """
        Generate the data readiness report and save it to the output directory.
        """
        output_dir = self.server_agent_config.client_configs.data_readiness_configs.get("output_dirname", "./output")
        output_filename = self.server_agent_config.client_configs.data_readiness_configs.get("output_filename", "data_readiness_report")

        if not os.path.exists(output_dir):
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Save JSON report
        json_file_path = get_unique_file_path(output_dir, output_filename, "json")
        save_json_report(json_file_path, readiness_report, self.logger)

        # Generate and save HTML report
        html_file_path = get_unique_file_path(output_dir, output_filename, "html")
        html_content = generate_html_content(readiness_report)
        save_html_report(html_file_path, html_content, self.logger)

        self._data_readiness_reports = {}

    def server_terminated(self):
        """Indicate whether the server can be terminated from listening to the clients."""
        if not hasattr(self, "closed_clients"):
            return False
        with self._close_connection_lock:
            terminated = len(self.closed_clients) >= self.get_num_clients()
        if terminated:
            self.clean_up()
        return terminated
    
    def clean_up(self) -> None:
        """
        Nececessary clean-up operations.
        No need to call this method if using `server_terminated` to check the termination status.
        """
        if not hasattr(self, "cleaned"):
            self.cleaned = False
        if not self.cleaned:
            self.cleaned = True
            if hasattr(self.scheduler, "clean_up"):
                self.scheduler.clean_up()

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
        if hasattr(self.server_agent_config.client_configs, "model_configs"):
            self._set_seed()
            model_configs = (
                self.server_agent_config.client_configs.model_configs 
                if hasattr(self.server_agent_config.client_configs, "model_configs") 
                else self.server_agent_config.server_configs.model_configs
            )
            self.model = create_instance_from_file(
                model_configs.model_path,
                model_configs.model_name,
                **(model_configs.model_kwargs if hasattr(model_configs, "model_kwargs") else {})
            )
            # load the model source file and delete model path
            if hasattr(self.server_agent_config.client_configs, "model_configs"):
                with open(model_configs.model_path, 'r') as f:
                    self.server_agent_config.client_configs.model_configs.model_source = f.read()
                del self.server_agent_config.client_configs.model_configs.model_path
        else:
            self.model = None

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

    def _load_scheduler(self) -> None:
        """Obtain the scheduler."""
        if hasattr(self.server_agent_config.server_configs, "aggregator_path"):
            # Load the user-defined aggregator from the file
            self.aggregator = create_instance_from_file(
                self.server_agent_config.server_configs.aggregator_path,
                self.server_agent_config.server_configs.aggregator,
                aggregator_configs=OmegaConf.create(
                    self.server_agent_config.server_configs.aggregator_kwargs if
                    hasattr(self.server_agent_config.server_configs, "aggregator_kwargs") else {}
                ),
                logger=self.logger,
            )
        else:
            self.aggregator: BaseAggregator = eval(self.server_agent_config.server_configs.aggregator)(
                self.model,
                OmegaConf.create(
                    self.server_agent_config.server_configs.aggregator_kwargs if
                    hasattr(self.server_agent_config.server_configs, "aggregator_kwargs") else {}
                ),
                self.logger,
            )
        
        if hasattr(self.server_agent_config.server_configs, "scheduler_path"):
            # Load the user-defined scheduler from the file
            self.scheduler = create_instance_from_file(
                self.server_agent_config.server_configs.scheduler_path,
                self.server_agent_config.server_configs.scheduler,
                scheduler_configs=OmegaConf.create(
                    self.server_agent_config.server_configs.scheduler_kwargs if 
                    hasattr(self.server_agent_config.server_configs, "scheduler_kwargs") else {}
                ),
                aggregator=self.aggregator,
                logger=self.logger,
            )
        else:
            self.scheduler: BaseScheduler = eval(self.server_agent_config.server_configs.scheduler)(
                OmegaConf.create(
                    self.server_agent_config.server_configs.scheduler_kwargs if 
                    hasattr(self.server_agent_config.server_configs, "scheduler_kwargs") else {}
                ),
                self.aggregator,
                self.logger,
            )
            
    def _load_trainer(self) -> None:
        """
        Process the trainer configurations if the trainer is provided locally as a user-defined class.
        """
        if hasattr(self.server_agent_config.client_configs.train_configs, "trainer_path"):
            with open(self.server_agent_config.client_configs.train_configs.trainer_path, 'r') as f:
                self.server_agent_config.client_configs.train_configs.trainer_source = f.read()
            del self.server_agent_config.client_configs.train_configs.trainer_path

    def _load_compressor(self) -> None:
        """Obtain the compressor."""
        self.compressor = None
        self.enable_compression = False
        if not hasattr(self.server_agent_config.server_configs, "comm_configs"):
            return
        if not hasattr(self.server_agent_config.server_configs.comm_configs, "compressor_configs"):
            return
        if getattr(self.server_agent_config.server_configs.comm_configs.compressor_configs, "enable_compression", False):
            self.enable_compression = True
            self.compressor = eval(self.server_agent_config.server_configs.comm_configs.compressor_configs.lossy_compressor)(
                self.server_agent_config.server_configs.comm_configs.compressor_configs
            )

    def _bytes_to_model(self, model_bytes: bytes) -> Union[Dict, OrderedDict]:
        """Deserialize the model from bytes."""
        if not self.enable_compression:
            return torch.load(io.BytesIO(model_bytes))
        else:
            if self.model is None:
                raise ValueError("Model is not provided to the server, so you cannot use compression.")
            return self.compressor.decompress_model(model_bytes, self.model)
            
    def _load_val_data(self) -> None:
        if hasattr(self.server_agent_config.server_configs, "val_data_configs"):
            self._val_dataset = run_function_from_file(
                self.server_agent_config.server_configs.val_data_configs.dataset_path,
                self.server_agent_config.server_configs.val_data_configs.dataset_name,
                **(
                    self.server_agent_config.server_configs.val_data_configs.dataset_kwargs
                    if hasattr(self.server_agent_config.server_configs.val_data_configs, "dataset_kwargs")
                    else {}
                )
            )
            self._val_dataloader = DataLoader(
                self._val_dataset,
                batch_size=self.server_agent_config.server_configs.val_data_configs.get("batch_size", 1),
                shuffle=self.server_agent_config.server_configs.val_data_configs.get("shuffle", False),
                num_workers=self.server_agent_config.server_configs.val_data_configs.get("num_workers", 0),
            )
    
    def _validate(self) -> Tuple[float, float]:
        """
        Validate the model
        :return: loss, accuracy
        """
        device = self.server_agent_config.server_configs.get("device", "cpu")
        self.model.to(device)
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            target_pred, target_true = [], []
            for data, target in self._val_dataloader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                val_loss += self.loss_fn(output, target).item()
                target_true.append(target.detach().cpu().numpy())
                target_pred.append(output.detach().cpu().numpy())
        val_loss /= len(self._val_dataloader)
        val_accuracy = float(self.metric(np.concatenate(target_true), np.concatenate(target_pred)))
        # Move the model back to the cpu for future aggregation
        if device != 'cpu':
            self.model.to('cpu')
        return val_loss, val_accuracy
    
    def _set_seed(self):
        """
        This function makes sure that all clients have the same initial model parameters.
        """
        seed_value = self.server_agent_config.client_configs.model_configs.get("seed", 42)
        self.logger.info(f"Setting seed value to {seed_value}")
        torch.manual_seed(seed_value)  # Set PyTorch seed
        torch.cuda.manual_seed_all(seed_value)  # Set seed for all GPUs
        torch.backends.cudnn.deterministic = True  # Use deterministic algorithms
        torch.backends.cudnn.benchmark = False  # Disable this to ensure reproducibility