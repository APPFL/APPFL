import time
import logging
from typing import Any, Dict, Optional
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader
from appfl.algorithm.trainer.base_trainer import BaseTrainer

try:
    import lightning.pytorch as pl
    from lightning.pytorch.loggers import MLFlowLogger
except ImportError:  # standalone `pytorch_lightning` package
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import MLFlowLogger

logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


class _RoundMLFlowLogger(MLFlowLogger):
    """
    MLflow logger shared by all rounds of one client. A new `pl.Trainer` is
    created every round, so Lightning's step counter restarts from zero; this
    logger shifts each round's steps by the steps of all previous rounds.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_offset = 0

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        if step is not None:
            step += self.step_offset
        super().log_metrics(metrics, step)


class LightningTrainer(BaseTrainer):
    """
    LightningTrainer:
        Trainer for FL clients whose model is a PyTorch Lightning `LightningModule`.
        Each round of local training is one call to `pl.Trainer.fit`, so the module's
        `training_step`, `validation_step` and `configure_optimizers` are used.

        Supported `train_configs`:
        - `mode`: "epoch" (uses `num_local_epochs` for how many epochs to train) or "step" (uses `num_local_steps`)
        - `device`: "cpu", "cuda", "cuda:<index>" or "mps"
        - `train_batch_size`, `val_batch_size`, `train_data_shuffle`, `num_workers`
        - `do_validation`: run the module's `validation_step` at the end of each local epoch
        - `do_pre_validation`: run the module's `validation_step` on the received global model before local training.
        - `mlflow_configs`: optional MLflow logging, with keys `enable_mlflow`,
          `tracking_uri`, `experiment_name` and `run_name` (defaults to the client id).
    """

    def __init__(
        self,
        model: "pl.LightningModule",
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        train_configs: DictConfig = DictConfig({}),
        logger: Optional[Any] = None,
        client_id: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_configs=train_configs,
            logger=logger,
            client_id=client_id,
        )
        assert isinstance(self.model, pl.LightningModule), (
            f"LightningTrainer requires a {pl.__name__}.LightningModule, got {type(self.model).__name__}"
        )
        self._sanity_check()

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.train_configs.get("train_batch_size", 32),
            shuffle=self.train_configs.get("train_data_shuffle", True),
            num_workers=self.train_configs.get("num_workers", 0),
        )
        self.val_dataloader = (
            DataLoader(
                self.val_dataset,
                batch_size=self.train_configs.get("val_batch_size", 32),
                shuffle=False,
                num_workers=self.train_configs.get("num_workers", 0),
            )
            if self.val_dataset is not None
            else None
        )
        self.accelerator, self.devices = self._parse_device(
            self.train_configs.get("device", "cpu")
        )

        mlflow_configs = self.train_configs.get("mlflow_configs", {})
        self.mlflow_logger = (
            _RoundMLFlowLogger(
                experiment_name=mlflow_configs.get("experiment_name", "appfl"),
                run_name=mlflow_configs.get("run_name", None)
                or str(self.client_id or "appfl-client"),
                tracking_uri=mlflow_configs.get("tracking_uri", None),
            )
            if mlflow_configs.get("enable_mlflow", False)
            else None
        )
        self.metric_names = None

    def train(self, **kwargs):
        if "round" in kwargs:
            self.round = kwargs["round"]
        start_time = time.time()
        self.val_results = {"round": self.round + 1}

        # Evaluate the received global model before local training
        if (
            self.train_configs.get("do_pre_validation", False)
            and self.val_dataloader is not None
        ):
            results = self._new_pl_trainer(logger=False).validate(
                model=self.model, dataloaders=self.val_dataloader, verbose=False
            )
            pre_val_results = {f"pre_{k}": float(v) for k, v in results[0].items()}
            self.val_results.update(pre_val_results)
            if self.mlflow_logger is not None:
                self.mlflow_logger.log_metrics(pre_val_results, step=0)

        if self.train_configs.mode == "epoch":
            budget = {"max_epochs": self.train_configs.num_local_epochs}
        else:
            budget = {"max_epochs": -1, "max_steps": self.train_configs.num_local_steps}
        trainer = self._new_pl_trainer(
            logger=self.mlflow_logger if self.mlflow_logger is not None else False,
            **budget,
        )
        trainer.fit(
            model=self.model,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=(
                self.val_dataloader
                if self.train_configs.get("do_validation", False)
                else None
            ),
        )

        self.val_results.update(
            {k: float(v) for k, v in trainer.callback_metrics.items()}
        )
        if self.mlflow_logger is not None:
            self.mlflow_logger.log_metrics(
                {"round": self.round + 1}, step=trainer.global_step
            )
            self.mlflow_logger.step_offset += trainer.global_step

        # Log a row per round with every metric the module logged via `self.log`
        if self.metric_names is None:
            self.metric_names = sorted(k for k in self.val_results if k != "round")
            self.logger.log_title(["Round", "Time"] + self.metric_names)
        self.logger.log_content(
            [self.round, round(time.time() - start_time, 2)]
            + [self.val_results.get(k, "") for k in self.metric_names]
        )
        self.round += 1

    def get_parameters(self):
        model_state = {
            k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
        }
        return (
            (model_state, self.val_results)
            if hasattr(self, "val_results")
            else model_state
        )

    def _new_pl_trainer(self, logger, **kwargs):
        return pl.Trainer(
            accelerator=self.accelerator,
            devices=self.devices,
            logger=logger,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            default_root_dir=self.train_configs.get(
                "logging_output_dirname", "./output"
            ),
            **kwargs,
        )

    def _sanity_check(self):
        assert self.train_configs.get("mode", None) in ["epoch", "step"], (
            "Training mode must be either 'epoch' or 'step'"
        )
        if self.train_configs.mode == "epoch":
            assert hasattr(self.train_configs, "num_local_epochs"), (
                "Number of local epochs must be specified"
            )
        else:
            assert hasattr(self.train_configs, "num_local_steps"), (
                "Number of local steps must be specified"
            )
        # These options change what the client sends or how it trains; fail
        # loudly instead of silently sending plain weights.
        for option in ("send_gradient", "use_secure_agg", "use_dp"):
            assert not self.train_configs.get(option, False), (
                f"{option}=True is not supported by LightningTrainer yet"
            )

    @staticmethod
    def _parse_device(device: str):
        """Map an APPFL device string to Lightning's `accelerator` and `devices`."""
        device = device.strip().lower()
        if device == "cpu":
            return "cpu", 1
        if device == "cuda":
            return "gpu", 1
        if device.startswith("cuda:"):
            return "gpu", [int(device.split(":", 1)[1])]
        if device == "mps":
            return "mps", 1
        raise ValueError(
            f"Unsupported device '{device}' for LightningTrainer, expected 'cpu', 'cuda', 'cuda:<index>' or 'mps'"
        )
