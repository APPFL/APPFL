from __future__ import annotations

import gc
import uuid
import torch
import importlib
from appfl.sim.algorithm.trainer import BaseTrainer
from omegaconf import DictConfig, OmegaConf
from typing import OrderedDict, Any
from appfl.sim.logger import ClientAgentFileLogger
from appfl.sim.misc.config_utils import (
    _create_instance_from_file,
    _run_function_from_file,
    build_loss_from_train_cfg,
)


class _NullClientLogger:
    """No-op logger used when client-side logging is disabled."""

    def log_title(self, titles):
        return None

    def log_content(self, contents):
        return None

    def info(self, info: str):
        return None

    def debug(self, debug: str):
        return None

    def error(self, error: str):
        return None

    def warning(self, warning: str):
        return None


try:
    import wandb
except Exception:  # pragma: no cover

    class _WandbStub:
        run = None

        class util:
            @staticmethod
            def generate_id():
                return str(uuid.uuid4())

        @staticmethod
        def init(*args, **kwargs):
            return None

        @staticmethod
        def log(*args, **kwargs):
            return None

    wandb = _WandbStub()


class ClientAgent:
    """
    The `ClientAgent` should act on behalf of the FL client to:
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
        self, client_agent_config: DictConfig | dict[str, Any] | None = None, **kwargs
    ) -> None:
        del kwargs
        if client_agent_config is None:
            self.client_agent_config = self._default_config()
        elif isinstance(client_agent_config, DictConfig):
            self.client_agent_config = client_agent_config
        else:
            self.client_agent_config = OmegaConf.create(client_agent_config)
        if "client_id" in self.client_agent_config:
            self.client_id: str | None = str(self.client_agent_config.client_id)
        else:
            self.client_id = str(uuid.uuid4())
        self.logger = None
        self.model = None
        self.loss_fn = None
        self.metric = None
        self.trainer = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._ensure_config_contract()
        self.optimize_memory = bool(
            self.client_agent_config.get("optimize_memory", True)
        )
        self._create_logger()
        self._init_wandb()
        self._load_model()
        self._load_loss()
        self._load_metric()
        self._load_data()
        self._load_trainer()

    def _ensure_config_contract(self) -> None:
        missing = [
            name
            for name in ("train_configs", "model_configs", "data_configs")
            if name not in self.client_agent_config
        ]
        if missing:
            raise ValueError(
                f"ClientAgentConfig is missing required sections: {', '.join(missing)}"
            )
        for name in ("train_configs", "model_configs", "data_configs"):
            if self.client_agent_config.get(name) is None:
                raise ValueError(f"ClientAgentConfig.{name} must not be None.")
        train_cfg = self.client_agent_config.train_configs
        if "trainer" not in train_cfg:
            raise ValueError("ClientAgentConfig.train_configs.trainer is required.")

    def train(self, **kwargs):
        """Train the model locally."""
        result = self.trainer.train(**kwargs)

        # Memory optimization: Garbage collection after training
        if self.optimize_memory:
            gc.collect()
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return result

    def get_parameters(
        self,
    ) -> dict | OrderedDict | bytes | tuple[dict | OrderedDict | bytes, dict]:
        """Return parameters for communication"""
        params = self.trainer.get_parameters()
        if isinstance(params, tuple):
            params, metadata = params
        else:
            metadata = None
        # Compression path is intentionally disabled in appfl[sim].
        # Memory optimization: Final cleanup
        if self.optimize_memory:
            gc.collect()

        return params if metadata is None else (params, metadata)

    def load_parameters(self, params) -> None:
        """Load parameters from the server."""
        self.trainer.load_parameters(params)

        # Memory optimization: Garbage collection after parameter loading
        if self.optimize_memory:
            gc.collect()

    def evaluate(self, split: str = "test", **kwargs) -> dict:
        if self.trainer is None:
            return {"loss": -1.0, "num_examples": 0, "metrics": {}}
        if not hasattr(self.trainer, "evaluate"):
            raise AttributeError("Trainer does not implement evaluate().")
        return self.trainer.evaluate(split=split, **kwargs)

    def _create_logger(self):
        """
        Create logger for the client agent to log local training process.
        You can modify or overwrite this method to create your own logger.
        """
        if self.logger is not None:
            return
        kwargs = {}
        kwargs["logging_id"] = self.client_id
        train_cfg = self.client_agent_config.train_configs
        if not train_cfg.get("client_logging_enabled", True):
            self.logger = _NullClientLogger()
            return
        kwargs["file_dir"] = train_cfg.get("logging_output_dirname", "./logs")
        kwargs["file_name"] = train_cfg.get("logging_output_filename", "log")
        self.logger = ClientAgentFileLogger(**kwargs)

    def _load_data(self) -> None:
        """Get train and validation dataloaders from local dataloader file."""
        if self.train_dataset is not None:
            return
        data_cfg = self.client_agent_config.data_configs
        if "dataset_path" not in data_cfg:
            self.train_dataset, self.val_dataset = None, None
            return
        self.train_dataset, self.val_dataset = _run_function_from_file(
            data_cfg.dataset_path,
            data_cfg.get("dataset_name", None),
            **data_cfg.get("dataset_kwargs", {}),
        )

    def _load_model(self) -> None:
        """
        Load model from `model_configs.model_path` (optional in simulator wiring).
        """
        if self.model is not None:
            return
        model_cfg = self.client_agent_config.model_configs
        if "model_path" in model_cfg:
            kwargs = model_cfg.get("model_kwargs", {})
            if "model_name" in model_cfg:
                self.model = _create_instance_from_file(
                    model_cfg.model_path,
                    model_cfg.model_name,
                    **kwargs,
                )
            else:
                self.model = _run_function_from_file(
                    model_cfg.model_path, None, **kwargs
                )
        else:
            self.model = None

    def _load_loss(self) -> None:
        """
        Load loss function from `train_configs`.
        """
        if self.loss_fn is not None:
            return
        train_cfg = self.client_agent_config.train_configs
        self.loss_fn = build_loss_from_train_cfg(train_cfg)

    def _load_metric(self) -> None:
        """
        Custom metric function loading is disabled in appfl-sim agent contract.
        """
        if self.metric is not None:
            return
        self.metric = None

    def _load_trainer(self) -> None:
        """Obtain a local trainer"""
        if self.trainer is not None:
            return
        if (
            self.train_dataset is None
            and self.val_dataset is None
            and self.test_dataset is None
        ):
            return
        train_cfg = self.client_agent_config.train_configs
        trainer_module = importlib.import_module("appfl.sim.algorithm.trainer")
        if not hasattr(trainer_module, train_cfg.trainer):
            if train_cfg.trainer == "MonaiTrainer":
                raise ImportError(
                    'Monai is not installed. Please install Monai to use MonaiTrainer using: pip install "appfl[monai]" or pip install -e ".[monai]" if installing from source.'
                )
            raise ValueError(f"Invalid trainer name: {train_cfg.trainer}")
        self.trainer: BaseTrainer = getattr(trainer_module, train_cfg.trainer)(
            model=self.model,
            loss_fn=self.loss_fn,
            metric=self.metric,
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            test_dataset=getattr(self, "test_dataset", None),
            train_configs=train_cfg,
            logger=self.logger,
            client_id=self.client_id,
        )

    def _init_wandb(self) -> None:
        """
        Initialize Weights and Biases for logging.
        """
        train_cfg = self.client_agent_config.train_configs
        train_cfg.enable_wandb = wandb.run is not None
        train_cfg.wandb_logging_id = self.client_id

    @staticmethod
    def _default_config() -> DictConfig:
        return OmegaConf.create(
            {
                "optimize_memory": True,
                "train_configs": {
                    "trainer": "FedavgTrainer",
                },
                "model_configs": {},
                "data_configs": {},
            }
        )

    @property
    def runtime_context(self):
        return getattr(self.trainer, "runtime_context", None)

    @runtime_context.setter
    def runtime_context(self, ctx):
        self.trainer.runtime_context = ctx
