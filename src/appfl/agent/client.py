import os
import uuid
import torch
import pathlib
import importlib
import torch.nn as nn
from datetime import datetime
from appfl.compressor import *
from appfl.config import ClientAgentConfig
from appfl.algorithm.trainer import BaseTrainer
from omegaconf import DictConfig, OmegaConf
from typing import Union, Dict, OrderedDict, Tuple, Optional
from appfl.misc.data_readiness import *
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
    
    :param client_agent_config: configurations for the client agent
    """
    def __init__(
        self, 
        client_agent_config: ClientAgentConfig = ClientAgentConfig(),
        **kwargs
    ) -> None:
        self.client_agent_config = client_agent_config
        self._create_logger()
        self._load_model()
        self._load_loss()
        self._load_metric()
        self._load_data()
        self._load_trainer()
        self._load_compressor()
        self.specified_metrics = None

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
        if isinstance(params, tuple):
            params, metadata = params
        else:
            metadata = None
        if self.enable_compression:
            params = self.compressor.compress_model(params)
        return params if metadata is None else (params, metadata)
    
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
            pathlib.Path(os.path.dirname(checkpoint_path)).mkdir(parents=True, exist_ok=True)
            
        torch.save(self.model.state_dict(), checkpoint_path)

    def generate_readiness_report(self, client_config):
        
        """
        Generate data readiness report based on the configuration provided by the server.
        """
        if hasattr(client_config.data_readiness_configs, "dr_metrics"):
            results = {}
            plot_results = {"plots": {}}
            to_combine_results = {"to_combine":{}}

            # Determine how to retrieve data input and labels based on dataset attributes
            if hasattr(self.train_dataset, 'data_label'):
                data_labels = self.train_dataset.data_label.tolist()
            else:
                try:

                    data_labels = [label.item() for _, label in self.train_dataset]
                except:

                    data_labels = [label for _, label in self.train_dataset]

            if hasattr(self.train_dataset, 'data_input'):
                data_input = self.train_dataset.data_input
            else:
                data_input = torch.stack([input_data for input_data, _ in self.train_dataset])

            fairness_feature_idx = getattr(client_config.data_readiness_configs, "fairness_feature_idx", None)

            # data_input, data_labels = balance_data(data_input, data_labels)
            # data_input, explained_variance = apply_pca(data_input)
            # data_input = normalize_data(data_input)

            # Define metrics with corresponding computation functions
            standard_metrics = {
                
                "class_imbalance": lambda: round(imbalance_degree(data_labels), 2),
                "sample_size": lambda: len(data_labels),
                "num_classes": lambda: len(set(data_labels)),
                "data_shape": lambda: (len(data_input), *data_input[0].size()),
                "completeness": lambda: completeness(data_input),
                "data_range": lambda: get_data_range(data_input),
                "overall_sparsity": lambda: sparsity(data_input),
                "variance": lambda: variance(data_input),
                "skewness": lambda: skewness(data_input),
                "entropy": lambda: entropy(data_input),
                "kurtosis": lambda: kurtosis(data_input),
                "class_distribution": lambda: class_distribution(data_labels),
                "brisque": lambda: brisque(data_input),
                "total_variation": lambda: total_variation(data_input),
                "sharpness": lambda: dataset_sharpness(data_input),
                "outlier_proportion": lambda: calculate_outlier_proportion(data_input),
                "time_to_event_imbalance": lambda: quantify_time_to_event_imbalance(data_labels)

            }

            plots = {
            "class_distribution_plot": lambda: plot_class_distribution(data_labels),
            "data_sample_plot": lambda: plot_data_sample(data_input),
            "data_distribution_plot": lambda: plot_data_distribution(data_input),
            "class_variance_plot": lambda: plot_class_variance(data_input, data_labels),
            "outlier_detection_plot": lambda: plot_outliers(data_input),
            "feature_correlation_plot": lambda: plot_feature_correlations(data_input),
            "feature_statistics_plot": lambda: plot_feature_statistics(data_input),
            "representative_rates_plot": lambda: plot_representative_rates(data_input, fairness_feature_idx),
            "feature_importance_plot": lambda: plot_feature_importance(data_input, data_labels),
            "incompleteness_plot": lambda: plot_incompleteness(data_input),
            "segmentation_class_distribution_plot": lambda: plot_segmentation_class_distribution(data_labels),
        }
            combine= {
                "feature_space_distribution": lambda: get_feature_space_distribution(data_input),
            }

            # Handle standard metrics
            for metric_name, compute_function in standard_metrics.items():

                if metric_name in client_config.data_readiness_configs.dr_metrics:
                    if getattr(client_config.data_readiness_configs.dr_metrics, metric_name):
                        results[metric_name] = compute_function()

            # Handle plot-specific metrics
            for metric_name, compute_function in plots.items():
                if metric_name in client_config.data_readiness_configs.dr_metrics.plot:
                    if getattr(client_config.data_readiness_configs.dr_metrics.plot, metric_name):
                        plot_results['plots'][metric_name] = compute_function()

            # Combine results with plot results
            results.update(plot_results)

            # Handle combined metrics
            for metric_name, compute_function in combine.items():
                if metric_name in client_config.data_readiness_configs.dr_metrics.combine:
                    if getattr(client_config.data_readiness_configs.dr_metrics.combine, metric_name):
                        to_combine_results['to_combine'][metric_name] = compute_function()
            
            results.update(to_combine_results)

            if hasattr(client_config.data_readiness_configs.dr_metrics, "specified_metrics") and hasattr(client_config.data_readiness_configs.dr_metrics.specified_metrics, "metric_path") and hasattr(client_config.data_readiness_configs.dr_metrics.specified_metrics, "metric_name"):
                self.specified_metrics = create_instance_from_file(
                    client_config.data_readiness_configs.dr_metrics.specified_metrics.metric_path,
                    client_config.data_readiness_configs.dr_metrics.specified_metrics.metric_name,
                    self.train_dataset
                )
                results['specified_metrics'] = self.specified_metrics.metric()
            
            # if hasattr(client_config.data_readiness_configs.dr_metrics.specified_metrics, "adapt_data") and client_config.data_readiness_configs.dr_metrics.specified_metrics.adapt_data:
            #     self.train_dataset = self.specified_metrics.rule(list(results['specified_metrics'].values())[0])
            #     self.logger.info(f"Data modified based on user-defined modification")

            return results
        else:
            return "Data readiness metrics not available in configuration"
        
    def adapt_data(self):
        """
        Modify the data based on the configuration provided by the server configs.
        """
        
        self.train_dataset = self.specified_metrics.remedy(self.specified_metrics.metric(), self.logger)
        
    # def adapt_data(self, client_config, metric_val):
    #     """
    #     Modify the data based on the configuration provided by the server configs.
    #     """

    #     if hasattr(client_config.data_readiness_configs.dr_metrics.specified_metrics, "threshold") and hasattr(client_config.data_readiness_configs.dr_metrics.specified_metrics, "proportion"):
    #         threshold = client_config.data_readiness_configs.dr_metrics.specified_metrics.threshold
    #         proportion = client_config.data_readiness_configs.dr_metrics.specified_metrics.proportion
            
    #         if metric_val > threshold:
    #             num_samples = int(proportion * len(self.train_dataset))
    #             self.train_dataset = random.sample(self.train_dataset, num_samples)
    #             self.logger.info(f"Data modified based on user-defined modification")

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
                # Convert target to Long if it is not already
            # self.train_dataset = apply_pca_to_dataset(self.train_dataset)
            # self.val_dataset = apply_pca_to_dataset(self.val_dataset)

            # self.train_dataset = normalize_dataset(self.train_dataset)
            # self.val_dataset = normalize_dataset(self.val_dataset)

            # self.train_dataset = balance_classes_undersample(self.train_dataset)

    def _load_model(self) -> None:
        """
        Load model from various sources with optional keyword arguments `model_kwargs`:
        - `model_path` and `model_name`: load model from a local file (usually for local simulation)
        - `model_source` and `model_name`: load model from a raw file source string (usually sent from the server)
        - Users can define their own way to load the model from other sources
        """
        if hasattr(self, "model") and self.model is not None:
            return
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
        if hasattr(self, "loss_fn") and self.loss_fn is not None:
            return
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
        if hasattr(self, "metric") and self.metric is not None:
            return
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
        if hasattr(self, "trainer") and self.trainer is not None:
            return
        if not hasattr(self.client_agent_config, "train_configs"):
            self.trainer = None
            return
        if (
            not hasattr(self.client_agent_config.train_configs, "trainer")
            and
            not hasattr(self.client_agent_config.train_configs, "trainer_path")
            and
            not hasattr(self.client_agent_config.train_configs, "trainer_source")
        ):
            self.trainer = None
            return
        if hasattr(self.client_agent_config.train_configs, "trainer_path"):
            self.trainer = create_instance_from_file(
                self.client_agent_config.train_configs.trainer_path,
                self.client_agent_config.train_configs.trainer,
                model=self.model,
                loss_fn=self.loss_fn,
                metric=self.metric,
                train_dataset=self.train_dataset,
                val_dataset=self.val_dataset,
                train_configs=self.client_agent_config.train_configs,
                logger=self.logger,
            )
        elif hasattr(self.client_agent_config.train_configs, "trainer_source"):
            self.trainer = create_instance_from_file_source(
                self.client_agent_config.train_configs.trainer_source,
                self.client_agent_config.train_configs.trainer,
                model=self.model,
                loss_fn=self.loss_fn,
                metric=self.metric,
                train_dataset=self.train_dataset,
                val_dataset=self.val_dataset,
                train_configs=self.client_agent_config.train_configs,
                logger=self.logger,
            )
        else:
            trainer_module = importlib.import_module('appfl.algorithm.trainer')
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
