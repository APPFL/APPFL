import copy
import time
import torch
import wandb
import importlib
import numpy as np
from torch.nn import Module
from omegaconf import DictConfig
from typing import Tuple, Dict, Optional, Any
from torch.utils.data import Dataset, DataLoader
from appfl.privacy import (
    laplace_mechanism_output_perturb,
    gaussian_mechanism_output_perturb,
    make_private_with_opacus,
)
from appfl.algorithm.trainer.base_trainer import BaseTrainer
from appfl.misc.utils import parse_device_str, apply_model_device
from appfl.misc.memory_utils import (
    extract_model_state_optimized,
    safe_inplace_operation,
    optimize_memory_cleanup,
)
from opacus.utils.batch_memory_manager import BatchMemoryManager
import logging

logging.getLogger().handlers.clear()
logging.getLogger().setLevel(logging.WARNING)


class VanillaTrainer(BaseTrainer):
    """
    VanillaTrainer:
        Vanilla trainer for FL clients, which trains the model using `torch.optim`
        optimizers for a certain number of local epochs or local steps.
        Users need to specify which training model to use in the configuration,
        as well as the number of local epochs or steps.
    """

    def __init__(
        self,
        model: Optional[Module] = None,
        loss_fn: Optional[Module] = None,
        metric: Optional[Any] = None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        train_configs: DictConfig = DictConfig({}),
        logger: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            metric=metric,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_configs=train_configs,
            logger=logger,
            **kwargs,
        )
        # Check for optimize_memory in train_configs, default to True
        self.optimize_memory = getattr(train_configs, "optimize_memory", True)

        self.privacy_engine = None
        if not hasattr(self.train_configs, "device"):
            self.train_configs.device = "cpu"
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
                shuffle=self.train_configs.get("val_data_shuffle", False),
                num_workers=self.train_configs.get("num_workers", 0),
            )
            if self.val_dataset is not None
            else None
        )
        if (
            hasattr(self.train_configs, "enable_wandb")
            and self.train_configs.enable_wandb
        ):
            self.enabled_wandb = True
            self.wandb_logging_id = self.train_configs.wandb_logging_id
        else:
            self.enabled_wandb = False
        self._sanity_check()

        # Extract train device, and configurations for possible DataParallel
        self.device_config, self.device = parse_device_str(self.train_configs.device)

    def train(self, **kwargs):
        """
        Train the model for a certain number of local epochs or steps and store the mode state
        (probably with perturbation for differential privacy) in `self.model_state`.
        """
        if "round" in kwargs:
            self.round = kwargs["round"]
        self.val_results = {"round": self.round + 1}

        # Store the previous model state for gradient computation
        send_gradient = self.train_configs.get("send_gradient", False)
        if send_gradient:
            if self.optimize_memory:
                self.model_prev = extract_model_state_optimized(
                    self.model, include_buffers=True, cpu_transfer=False
                )
            else:
                self.model_prev = copy.deepcopy(self.model.state_dict())

        # Configure model for possible DataParallel
        self.model = apply_model_device(self.model, self.device_config, self.device)

        do_validation = (
            self.train_configs.get("do_validation", False)
            and self.val_dataloader is not None
        )
        do_pre_validation = (
            self.train_configs.get("do_pre_validation", False)
            and self.val_dataloader is not None
        )

        # Set up logging title
        title = (
            ["Round", "Time", "Train Loss", "Train Accuracy"]
            if (not do_validation) and (not do_pre_validation)
            else (
                [
                    "Round",
                    "Pre Val?",
                    "Time",
                    "Train Loss",
                    "Train Accuracy",
                    "Val Loss",
                    "Val Accuracy",
                ]
                if do_pre_validation
                else [
                    "Round",
                    "Time",
                    "Train Loss",
                    "Train Accuracy",
                    "Val Loss",
                    "Val Accuracy",
                ]
            )
        )
        if self.train_configs.mode == "epoch":
            title.insert(1, "Epoch")

        if self.round == 0:
            self.logger.log_title(title)
        self.logger.set_title(title)

        if do_pre_validation:
            val_loss, val_accuracy = self._validate()
            self.val_results["pre_val_loss"] = val_loss
            self.val_results["pre_val_accuracy"] = val_accuracy
            content = [self.round, "Y", " ", " ", " ", val_loss, val_accuracy]
            if self.train_configs.mode == "epoch":
                content.insert(1, 0)
            self.logger.log_content(content)
            if self.enabled_wandb:
                wandb.log(
                    {
                        f"{self.wandb_logging_id}/val-loss (before train)": val_loss,
                        f"{self.wandb_logging_id}/val-accuracy (before train)": val_accuracy,
                    }
                )

        # Start training
        optim_module = importlib.import_module("torch.optim")
        assert hasattr(optim_module, self.train_configs.optim), (
            f"Optimizer {self.train_configs.optim} not found in torch.optim"
        )
        optimizer = getattr(optim_module, self.train_configs.optim)(
            self.model.parameters(), **self.train_configs.optim_args
        )

        if self.train_configs.get("use_dp", False) and (
            self.train_configs.get("dp_mechanism", "laplace") == "opacus"
        ):
            dp_cfg = self.train_configs.get("dp_config", {})
            noise_multiplier = dp_cfg.get("noise_multiplier", 1.0)
            max_grad_norm = dp_cfg.get("max_grad_norm", 1.0)

            self.model, optimizer, self.train_dataloader, self.privacy_engine = (
                make_private_with_opacus(
                    self.model,
                    optimizer,
                    self.train_dataloader,
                    noise_multiplier=noise_multiplier,
                    max_grad_norm=max_grad_norm,
                    device=self.train_configs.device,
                )
            )

        if self.train_configs.mode == "epoch":
            for epoch in range(self.train_configs.num_local_epochs):
                start_time = time.time()
                train_loss, target_true, target_pred = 0, [], []
                for data, target in self.train_dataloader:
                    loss, pred, label = self._train_batch(optimizer, data, target)
                    train_loss += loss
                    target_true.append(label)
                    target_pred.append(pred)
                train_loss /= len(self.train_dataloader)
                target_true, target_pred = (
                    np.concatenate(target_true),
                    np.concatenate(target_pred),
                )
                train_accuracy = float(self.metric(target_true, target_pred))
                if do_validation:
                    val_loss, val_accuracy = self._validate()
                    if "val_loss" not in self.val_results:
                        self.val_results["val_loss"] = []
                        self.val_results["val_accuracy"] = []
                    self.val_results["val_loss"].append(val_loss)
                    self.val_results["val_accuracy"].append(val_accuracy)
                per_epoch_time = time.time() - start_time
                if self.enabled_wandb:
                    wandb.log(
                        {
                            f"{self.wandb_logging_id}/train-loss (during train)": train_loss,
                            f"{self.wandb_logging_id}/train-accuracy (during train)": train_accuracy,
                            f"{self.wandb_logging_id}/val-loss (during train)": val_loss,
                            f"{self.wandb_logging_id}/val-accuracy (during train)": val_accuracy,
                        }
                    )
                self.logger.log_content(
                    [self.round, epoch, per_epoch_time, train_loss, train_accuracy]
                    if (not do_validation) and (not do_pre_validation)
                    else (
                        [
                            self.round,
                            epoch,
                            per_epoch_time,
                            train_loss,
                            train_accuracy,
                            val_loss,
                            val_accuracy,
                        ]
                        if not do_pre_validation
                        else [
                            self.round,
                            epoch,
                            "N",
                            per_epoch_time,
                            train_loss,
                            train_accuracy,
                            val_loss,
                            val_accuracy,
                        ]
                    )
                )
        else:
            start_time = time.time()
            train_loss, target_true, target_pred = 0, [], []
            if (
                self.train_configs.get("use_dp", False)
                and self.train_configs.get("dp_mechanism", "laplace") == "opacus"
            ):
                with BatchMemoryManager(
                    data_loader=self.train_dataloader,
                    max_physical_batch_size=self.train_configs.get(
                        "train_batch_size", 32
                    ),
                    optimizer=optimizer,
                ) as memory_safe_data_loader:
                    step_count = 0
                    for data, target in memory_safe_data_loader:
                        loss, pred, label = self._train_batch(optimizer, data, target)
                        train_loss += loss
                        target_true.append(label)
                        target_pred.append(pred)
                        step_count += 1
                        if step_count >= self.train_configs.num_local_steps:
                            break
            else:
                data_iter = iter(self.train_dataloader)
                for _ in range(self.train_configs.num_local_steps):
                    try:
                        data, target = next(data_iter)
                    except:  # noqa E722
                        data_iter = iter(self.train_dataloader)
                        data, target = next(data_iter)
                    loss, pred, label = self._train_batch(optimizer, data, target)
                    train_loss += loss
                    target_true.append(label)
                    target_pred.append(pred)
            train_loss /= len(self.train_dataloader)
            target_true, target_pred = (
                np.concatenate(target_true),
                np.concatenate(target_pred),
            )
            train_accuracy = float(self.metric(target_true, target_pred))
            if do_validation:
                val_loss, val_accuracy = self._validate()
                self.val_results["val_loss"] = val_loss
                self.val_results["val_accuracy"] = val_accuracy
            per_step_time = time.time() - start_time
            if self.enabled_wandb:
                wandb.log(
                    {
                        f"{self.wandb_logging_id}/train-loss (during train)": train_loss,
                        f"{self.wandb_logging_id}/train-accuracy (during train)": train_accuracy,
                        f"{self.wandb_logging_id}/val-loss (during train)": val_loss,
                        f"{self.wandb_logging_id}/val-accuracy (during train)": val_accuracy,
                    }
                )
            self.logger.log_content(
                [self.round, per_step_time, train_loss, train_accuracy]
                if (not do_validation) and (not do_pre_validation)
                else (
                    [
                        self.round,
                        per_step_time,
                        train_loss,
                        train_accuracy,
                        val_loss,
                        val_accuracy,
                    ]
                    if not do_pre_validation
                    else [
                        self.round,
                        "N",
                        per_step_time,
                        train_loss,
                        train_accuracy,
                        val_loss,
                        val_accuracy,
                    ]
                )
            )

        # --- Log DP budget ---
        # if self.train_configs.get("use_dp", False) and self.train_configs.get("dp_mechanism", "none") == "opacus":
        #     epsilon = self.privacy_engine.get_epsilon(delta=1e-5)
        #     print(f"[DP] Training completed with (ε = {epsilon:.2f}, δ = 1e-5)")

        # If model was wrapped in DataParallel, unload it
        if self.device_config["device_type"] == "gpu-multi":
            self.model = self.model.module.to(self.device)

        self.round += 1

        # Differential privacy
        if self.train_configs.get("use_dp", False) and (
            self.train_configs.get("dp_mechanism", "laplace") == "gaussian"
        ):
            assert hasattr(self.train_configs, "clip_value"), (
                "Using laplace differential privacy, and gradient clipping value must be specified"
            )
            assert hasattr(self.train_configs, "epsilon"), (
                "Using laplace differential privacy, and privacy budget (epsilon) must be specified"
            )
            sensitivity = (
                2.0 * self.train_configs.clip_value * self.train_configs.optim_args.lr
            )
            self.model_state = gaussian_mechanism_output_perturb(
                self.model,
                sensitivity,
                self.train_configs.epsilon,
            )
        elif self.train_configs.get("use_dp", False) and (
            self.train_configs.get("dp_mechanism", "laplace") == "laplace"
        ):
            assert hasattr(self.train_configs, "clip_value"), (
                "Using laplace differential privacy, and gradient clipping value must be specified"
            )
            assert hasattr(self.train_configs, "epsilon"), (
                "Using laplace differential privacy, and privacy budget (epsilon) must be specified"
            )
            sensitivity = (
                2.0 * self.train_configs.clip_value * self.train_configs.optim_args.lr
            )
            self.model_state = laplace_mechanism_output_perturb(
                self.model,
                sensitivity,
                self.train_configs.epsilon,
            )
        else:
            if self.optimize_memory:
                self.model_state = extract_model_state_optimized(
                    self.model, include_buffers=True, cpu_transfer=False
                )
            else:
                self.model_state = copy.deepcopy(self.model.state_dict())

        # Move to CPU for communication
        if "cuda" in self.train_configs.device:
            if self.optimize_memory:
                for k in self.model_state:
                    if self.model_state[k].device.type != "cpu":
                        self.model_state[k] = self.model_state[k].cpu()
                optimize_memory_cleanup(force_gc=True)
            else:
                for k in self.model_state:
                    self.model_state[k] = self.model_state[k].cpu()

        # Compute the gradient if needed
        if send_gradient:
            self._compute_gradient()

    def get_parameters(self) -> Dict:
        if not hasattr(self, "model_state"):
            if self.optimize_memory:
                self.model_state = extract_model_state_optimized(
                    self.model, include_buffers=True, cpu_transfer=False
                )
            else:
                self.model_state = copy.deepcopy(self.model.state_dict())
        return (
            (self.model_state, self.val_results)
            if hasattr(self, "val_results")
            else self.model_state
        )

    def _sanity_check(self):
        """
        Check if the configurations are valid.
        """
        assert hasattr(self.train_configs, "mode"), "Training mode must be specified"
        assert self.train_configs.mode in [
            "epoch",
            "step",
        ], "Training mode must be either 'epoch' or 'step'"
        if self.train_configs.mode == "epoch":
            assert hasattr(self.train_configs, "num_local_epochs"), (
                "Number of local epochs must be specified"
            )
        else:
            assert hasattr(self.train_configs, "num_local_steps"), (
                "Number of local steps must be specified"
            )

    def _validate(self) -> Tuple[float, float]:
        """
        Validate the model
        :return: loss, accuracy
        """
        device = self.device
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            target_pred, target_true = [], []
            for data, target in self.val_dataloader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                val_loss += self.loss_fn(output, target).item()
                target_true.append(target.detach().cpu().numpy())
                target_pred.append(output.detach().cpu().numpy())
        val_loss /= len(self.val_dataloader)
        val_accuracy = float(
            self.metric(np.concatenate(target_true), np.concatenate(target_pred))
        )
        self.model.train()
        return val_loss, val_accuracy

    def _train_batch(
        self, optimizer: torch.optim.Optimizer, data, target
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Train the model for one batch of data
        :param optimizer: torch optimizer
        :param data: input data
        :param target: target label
        :return: loss, prediction, label
        """
        device = self.device
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = self.model(data)
        loss = self.loss_fn(output, target)
        loss.backward()
        if getattr(self.train_configs, "clip_grad", False) or getattr(
            self.train_configs, "use_dp", False
        ):
            assert hasattr(self.train_configs, "clip_value"), (
                "Gradient clipping value must be specified"
            )
            assert hasattr(self.train_configs, "clip_norm"), (
                "Gradient clipping norm must be specified"
            )
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.train_configs.clip_value,
                norm_type=self.train_configs.clip_norm,
            )
        optimizer.step()
        return loss.item(), output.detach().cpu().numpy(), target.detach().cpu().numpy()

    def _compute_gradient(self) -> None:
        """
        Compute the gradient of the model and store in `self.model_state`,
        where gradient = prev_model - new_model
        """
        if not hasattr(self, "named_parameters"):
            self.named_parameters = set()
            for name, _ in self.model.named_parameters():
                self.named_parameters.add(name)

        if self.optimize_memory:
            with torch.no_grad():
                for name in self.model_state:
                    if name in self.named_parameters:
                        prev_param = (
                            self.model_prev[name].cpu()
                            if self.model_prev[name].device.type != "cpu"
                            else self.model_prev[name]
                        )
                        self.model_state[name] = safe_inplace_operation(
                            prev_param, "sub", self.model_state[name], alpha=1
                        )
            optimize_memory_cleanup(self.model_prev, force_gc=True)
            del self.model_prev
        else:
            for name in self.model_state:
                if name in self.named_parameters:
                    self.model_state[name] = (
                        self.model_prev[name].cpu() - self.model_state[name]
                    )
