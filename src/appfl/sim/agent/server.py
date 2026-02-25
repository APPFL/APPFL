import torch
from appfl.sim.logger import ServerAgentFileLogger
from appfl.sim.algorithm.scheduler import BaseScheduler
from appfl.sim.algorithm.aggregator import BaseAggregator
from appfl.sim.metrics import MetricsManager, parse_metric_names
from appfl.sim.misc.runtime_utils import (
    _create_aggregator_instance,
    _create_scheduler_instance,
)
from appfl.sim.misc.config_utils import (
    _create_instance_from_file,
    _run_function_from_file,
    build_loss_from_train_cfg,
)
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig
from typing import Union, Dict, OrderedDict, Optional, Any

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None


class ServerAgent:
    """
    FL server runtime for simulation: model/loss/scheduler wiring, aggregation, and evaluation.
    """

    def __init__(
        self, server_agent_config: Optional[DictConfig | Dict[str, Any]] = None
    ) -> None:
        if server_agent_config is None:
            self.server_agent_config = self._default_config()
        elif isinstance(server_agent_config, DictConfig):
            self.server_agent_config = server_agent_config
        else:
            self.server_agent_config = OmegaConf.create(server_agent_config)
        self.num_clients: Optional[int] = None
        self.model = None
        self.loss_fn = None
        self.aggregator = None
        self.scheduler = None
        self._eval_dataset = None
        self._eval_dataloader = None
        self._ensure_config_contract()
        self._set_num_clients()
        self._create_logger()
        self._load_model()
        self._load_loss()
        self._load_scheduler()
        self._load_eval_data()

    def _ensure_config_contract(self) -> None:
        if "server_configs" not in self.server_agent_config:
            raise ValueError(
                "ServerAgentConfig is missing required section: server_configs"
            )
        if "client_configs" not in self.server_agent_config:
            raise ValueError(
                "ServerAgentConfig is missing required section: client_configs"
            )
        client_cfg = self.server_agent_config.client_configs
        for name in ("train_configs", "model_configs"):
            if name not in client_cfg:
                raise ValueError(
                    f"ServerAgentConfig.client_configs is missing required section: {name}"
                )
            if client_cfg.get(name) is None:
                raise ValueError(
                    f"ServerAgentConfig.client_configs.{name} must not be None."
                )
        if self.server_agent_config.server_configs is None:
            raise ValueError("ServerAgentConfig.server_configs must not be None.")
        for name in ("num_clients", "aggregator", "scheduler"):
            if name not in self.server_agent_config.server_configs:
                raise ValueError(
                    f"ServerAgentConfig.server_configs.{name} is required."
                )

    def get_parameters(self, **kwargs):
        """
        Return the global model to the clients.
        """
        del kwargs
        return self.scheduler.get_parameters()

    def aggregate(
        self,
        local_states: Dict[Union[int, str], Union[Dict, OrderedDict]],
        sample_sizes: Dict[Union[int, str], int],
        client_train_stats: Optional[Dict[Union[int, str], Dict[str, Any]]] = None,
    ) -> Dict[Union[int, str], float]:
        """
        Aggregate local client updates using the configured APPFL aggregator.
        Returns normalized aggregation weights for logging.
        """
        if not local_states:
            return {}
        if self.aggregator is None:
            raise RuntimeError("ServerAgent aggregator is not initialized.")

        if (
            hasattr(self.aggregator, "model")
            and getattr(self.aggregator, "model", None) is None
            and self.model is not None
        ):
            self.aggregator.model = self.model

        total = float(sum(int(sample_sizes.get(cid, 0)) for cid in local_states))
        if total <= 0.0:
            weights = {cid: 1.0 / len(local_states) for cid in local_states}
        else:
            weights = {
                cid: float(int(sample_sizes.get(cid, 0))) / total
                for cid in local_states
            }

        for cid, size in sample_sizes.items():
            self.aggregator.set_client_sample_size(cid, int(size))

        aggregated = self.aggregator.aggregate(
            local_states,
            client_train_stats=client_train_stats or {},
        )
        if isinstance(aggregated, tuple):
            aggregated = aggregated[0]
        if isinstance(aggregated, dict) and self.model is not None:
            self.model.load_state_dict(aggregated, strict=False)
        return weights

    @torch.no_grad()
    def evaluate(self, round_idx: Optional[int] = None) -> Dict[str, Any]:
        return self._evaluate_metrics(round_idx=round_idx)

    def _evaluate_metrics(self, round_idx: Optional[int] = None) -> Dict[str, Any]:
        if self._eval_dataset is None:
            return {"loss": -1.0, "num_examples": 0, "metrics": {}}
        if len(self._eval_dataset) == 0:
            return {"loss": -1.0, "num_examples": 0, "metrics": {}}
        if self.model is None:
            return {"loss": -1.0, "num_examples": 0, "metrics": {}}

        if self.loss_fn is None:
            self.loss_fn = torch.nn.CrossEntropyLoss()

        if self._eval_dataloader is None:
            self._eval_dataloader = DataLoader(
                self._eval_dataset,
                batch_size=int(
                    self.server_agent_config.server_configs.get("eval_batch_size", 128)
                ),
                shuffle=False,
                num_workers=int(
                    self.server_agent_config.server_configs.get("num_workers", 0)
                ),
            )

        device = torch.device(
            str(self.server_agent_config.server_configs.get("device", "cpu"))
        )
        eval_metric_names = parse_metric_names(
            self.server_agent_config.server_configs.get(
                "eval_metrics",
                self.server_agent_config.client_configs.train_configs.get(
                    "eval_metrics", None
                ),
            )
        )
        manager = MetricsManager(eval_metrics=eval_metric_names)
        was_training = self.model.training
        self.model.to(device)
        if hasattr(self.loss_fn, "to"):
            self.loss_fn = self.loss_fn.to(device)
        self.model.eval()

        total_examples = 0
        show_progress = bool(
            self.server_agent_config.server_configs.get("eval_show_progress", True)
        )
        progress_bar = None
        iterator = self._eval_dataloader
        if show_progress and _tqdm is not None:
            try:
                total_batches = len(self._eval_dataloader)
            except Exception:
                total_batches = None
            if round_idx is None:
                desc = "appfl-sim: ✅[Server | Evaluation (Global)]"
            else:
                desc = f"appfl-sim: ✅[Server (Round {int(round_idx):04d}) Evaluation (Global)]"
            progress_bar = _tqdm(
                self._eval_dataloader,
                total=total_batches,
                desc=desc,
                leave=False,
                dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            )
            iterator = progress_bar

        try:
            for inputs, targets in iterator:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                logits = self.model(inputs)
                loss = self.loss_fn(logits, targets)
                logits_cpu = logits.detach().cpu()
                targets_cpu = targets.detach().cpu()

                manager.track(float(loss.item()), logits_cpu, targets_cpu)
                bs = targets_cpu.size(0)
                total_examples += bs
        finally:
            if progress_bar is not None:
                progress_bar.close()

        result = manager.aggregate(total_len=total_examples)

        if was_training:
            self.model.train()
        self.model.to("cpu")
        if hasattr(self.loss_fn, "to"):
            self.loss_fn = self.loss_fn.to("cpu")
        return result

    def _create_logger(self) -> None:
        kwargs = {}
        if (
            self.server_agent_config.server_configs.get("logging_output_dirname", None)
            is not None
        ):
            kwargs["file_dir"] = (
                self.server_agent_config.server_configs.logging_output_dirname
            )
        if (
            self.server_agent_config.server_configs.get("logging_output_filename", None)
            is not None
        ):
            kwargs["file_name"] = (
                self.server_agent_config.server_configs.logging_output_filename
            )
        self.logger = ServerAgentFileLogger(**kwargs)

    def _load_model(self) -> None:
        """
        Load model from the definition file, and read the source code of the model for sendind to the client.
        User can overwrite this method to load the model from other sources.
        """
        if self.model is not None:
            return
        self._set_seed()
        model_configs = self.server_agent_config.client_configs.model_configs
        if "model_path" not in model_configs:
            self.model = None
            return
        if "model_name" in model_configs:
            self.model = _create_instance_from_file(
                model_configs.model_path,
                model_configs.model_name,
                **model_configs.get("model_kwargs", {}),
            )
        else:
            self.model = _run_function_from_file(
                model_configs.model_path,
                None,
                **model_configs.get("model_kwargs", {}),
            )

    def _load_loss(self) -> None:
        """
        Load loss function from client train configuration.
        """
        train_cfg = self.server_agent_config.client_configs.train_configs
        self.loss_fn = build_loss_from_train_cfg(train_cfg)

    def _load_scheduler(self) -> None:
        """Obtain the scheduler."""
        server_cfg = self.server_agent_config.server_configs
        self.aggregator: BaseAggregator = _create_aggregator_instance(
            aggregator_name=server_cfg.aggregator,
            model=self.model,
            aggregator_config=OmegaConf.create(server_cfg.get("aggregator_kwargs", {})),
            logger=self.logger,
        )

        self.scheduler: BaseScheduler = _create_scheduler_instance(
            scheduler_name=server_cfg.scheduler,
            scheduler_config=OmegaConf.create(server_cfg.get("scheduler_kwargs", {})),
            aggregator=self.aggregator,
            logger=self.logger,
        )

    def _load_eval_data(self) -> None:
        if self._eval_dataset is None:
            self._eval_dataloader = None
            return
        self._eval_dataloader = DataLoader(
            self._eval_dataset,
            batch_size=int(
                self.server_agent_config.server_configs.get("eval_batch_size", 128)
            ),
            shuffle=False,
            num_workers=int(
                self.server_agent_config.server_configs.get("num_workers", 0)
            ),
        )

    def _set_seed(self):
        """
        This function makes sure that all clients have the same initial model parameters.
        """
        seed_value = self.server_agent_config.client_configs.model_configs.get(
            "seed", 42
        )
        torch.manual_seed(seed_value)  # Set PyTorch seed
        torch.cuda.manual_seed_all(seed_value)  # Set seed for all GPUs
        torch.backends.cudnn.deterministic = True  # Use deterministic algorithms
        torch.backends.cudnn.benchmark = False  # Disable this to ensure reproducibility

    def _set_num_clients(self) -> None:
        """
        Set the number of clients.
        The number of clients must be set in server_configs.
        """
        if self.num_clients is None:
            if "num_clients" not in self.server_agent_config.server_configs:
                raise ValueError("server_configs.num_clients is required.")
            self.num_clients = self.server_agent_config.server_configs.num_clients
            # Set num_clients for aggregator and scheduler
            if "scheduler_kwargs" not in self.server_agent_config.server_configs:
                self.server_agent_config.server_configs.scheduler_kwargs = (
                    OmegaConf.create({})
                )
            if "aggregator_kwargs" not in self.server_agent_config.server_configs:
                self.server_agent_config.server_configs.aggregator_kwargs = (
                    OmegaConf.create({})
                )
            self.server_agent_config.server_configs.scheduler_kwargs.num_clients = (
                self.num_clients
            )
            self.server_agent_config.server_configs.aggregator_kwargs.num_clients = (
                self.num_clients
            )
            # Set num_clients for server_configs
            self.server_agent_config.server_configs.num_clients = self.num_clients

    @staticmethod
    def _default_config() -> DictConfig:
        return OmegaConf.create(
            {
                "client_configs": {
                    "train_configs": {},
                    "model_configs": {},
                },
                "server_configs": {
                    "num_clients": 1,
                    "aggregator": "FedavgAggregator",
                    "aggregator_kwargs": {},
                    "scheduler": "FedavgScheduler",
                    "scheduler_kwargs": {},
                },
            }
        )
