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

class LLMTrainer_ds(VanillaTrainer):
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

    def train(self, **kwargs):
        """
        Train the model for a certain number of local epochs or steps and store the mode state
        (probably with perturbation for differential privacy) in `self.model_state`.
        """
        device = self.model_engine.device
        self.model_engine.train()
        self.total_pre_val_time = "n/a"
        self.total_val_time = "n/a"
        self.total_forward_time = 0.0
        self.total_backward_time = 0.0
        # Start training
        if self.train_configs.mode == "epoch":
            for epoch in range(self.train_configs.num_local_epochs):
                epoch_loss = 0
                for step, batch in enumerate(self.train_dataloader):
                    input_ids, labels = batch
                    input_ids = input_ids.squeeze(1).to(device)
                    labels = labels.squeeze(1).to(device)

                    forward_start_time = time.time()
                    outputs = self.model_engine(input_ids)
                    logits = outputs.logits

                    shifted_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
                    shifted_labels = labels[:, 1:].contiguous().view(-1)

                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(shifted_logits, shifted_labels)
                    self.total_forward_time += time.time() - forward_start_time

                    backward_start_time = time.time()
                    self.model_engine.backward(loss)
                    self.total_backward_time += time.time() - backward_start_time
                    
                    self.model_engine.step()

                    if self.global_rank == 0:
                        epoch_loss += loss.item()
                
            if self.global_rank == 0:
                avg_epoch_loss = epoch_loss / len(self.train_dataloader)
                print(f"average epoch loss is {avg_epoch_loss}")