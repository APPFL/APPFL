import copy
import time
import torch
import torch.nn as nn
import math
import wandb
import importlib
import numpy as np
from torch.nn import Module
from omegaconf import DictConfig
from typing import Tuple, Dict, Optional, Any
from torch.utils.data import Dataset, DataLoader
from appfl.privacy import laplace_mechanism_output_perturb
from appfl.algorithm.trainer.base_trainer import BaseTrainer
from appfl.misc.utils import parse_device_str, apply_model_device


from appfl.algorithm.trainer.vanilla_trainer import VanillaTrainer

class LLMTrainer(VanillaTrainer):
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
        if not hasattr(self.train_configs, "device"):
            self.train_configs.device = "cpu"
        
        def collate_fn(batch):
            input_ids = torch.stack([item[0] for item in batch])
            labels = torch.stack([item[1] for item in batch])

            return input_ids, labels

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.train_configs.get("train_batch_size", 32),
            shuffle=self.train_configs.get("train_data_shuffle", True),
            num_workers=self.train_configs.get("num_workers", 0),
            collate_fn=collate_fn
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
            pre_val_start_time = time.time()
            val_loss, val_accuracy = self._validate()
            self.total_pre_val_time = time.time() - pre_val_start_time
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
        if self.train_configs.mode == "epoch":
            for epoch in range(self.train_configs.num_local_epochs):
                start_time = time.time()
                train_loss = 0
                for step, batch in enumerate(self.train_dataloader):
                    input_ids, labels = batch
                    input_ids = input_ids.squeeze(1)
                    labels = labels.squeeze(1)
                    loss = self._train_batch(optimizer, input_ids, labels)
                    train_loss += loss
                train_loss /= len(self.train_dataloader)
                train_accuracy = "N/A"
                if do_validation:
                    val_start_time = time.time()
                    val_loss, val_accuracy = self._validate()
                    self.total_val_time = time.time() - val_start_time
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
            data_iter = iter(self.train_dataloader)
            train_loss, target_true, target_pred = 0, [], []
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

        # If model was wrapped in DataParallel, unload it
        if self.device_config["device_type"] == "gpu-multi":
            self.model = self.model.module.to(self.device)

        self.round += 1

        # Differential privacy
        if self.train_configs.get("use_dp", False):
            assert hasattr(self.train_configs, "clip_value"), (
                "Gradient clipping value must be specified"
            )
            assert hasattr(self.train_configs, "epsilon"), (
                "Privacy budget (epsilon) must be specified"
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
            self.model_state = copy.deepcopy(self.model.state_dict())

        # Move to CPU for communication
        if "cuda" in self.train_configs.device:
            for k in self.model_state:
                self.model_state[k] = self.model_state[k].cpu()

        # Compute the gradient if needed
        if send_gradient:
            self._compute_gradient()

    def _validate(self) -> Tuple[str, float]:
        """
        Validate the model
        :return: loss, accuracy
        """
        device = self.device
        self.model.eval()
        val_loss = "N/A"
        batch_size=self.train_configs.get("val_batch_size", 32)
        with torch.no_grad():
            val_input_ids = self.val_dataset.input_ids
            seq_len = self.model.seqlen

            total_tokens = val_input_ids.numel()
            nsamples = total_tokens // seq_len

            total_nll = 0.0

            for batch_start in range(0, nsamples, batch_size):
                batch_end = min(batch_start + batch_size, nsamples)
                batch_size = batch_end - batch_start

                batch_input = torch.zeros(
                    (batch_size, seq_len),
                    dtype=val_input_ids.dtype,
                    device=val_input_ids.device
                )

                for i in range(batch_start, batch_end):
                    batch_idx = i - batch_start
                    seq_start = i * seq_len
                    seq_end = (i+1) * seq_len
                    batch_input[batch_idx] = val_input_ids[0, seq_start:seq_end]
            
                batch_input = batch_input.to(device)

                outputs = self.model(batch_input)
                logits = outputs.logits
            
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = batch_input[:, 1:].contiguous()
            
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                token_losses = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), 
                    shift_labels.view(-1)
                )
            
                token_losses = token_losses.view(batch_size, -1)
                neg_log_likelihood = token_losses.sum(dim=1)

                total_nll += neg_log_likelihood.sum().item()
        
            # Calculate perplexity
            effective_tokens = nsamples * (seq_len - 1)
            avg_nll = total_nll / effective_tokens
            perplexity = math.exp(avg_nll)
        
        self.model.train()
        return val_loss, perplexity

    def _train_batch(
        self, optimizer: torch.optim.Optimizer, input_ids, labels
    ) -> Tuple[float]:
        """
        Train the model for one batch of data
        :param optimizer: torch optimizer
        :param data: input data
        :param target: target label
        :return: loss, prediction, label
        """
        device = self.device
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        forward_start_time = time.time()
        outputs = self.model(input_ids)
        logits = outputs.logits

        shifted_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
        shifted_labels = labels[:, 1:].contiguous().view(-1)

        loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(shifted_logits, shifted_labels)
        self.total_forward_time = time.time() - forward_start_time

        backward_start_time = time.time()
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
        self.total_backward_time = time.time() - backward_start_time
        return loss.item()