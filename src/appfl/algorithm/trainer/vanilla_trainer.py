import copy
import time
import torch
import importlib
import numpy as np
from torch.nn import Module
from omegaconf import DictConfig
from typing import Tuple, Dict, Optional, Any
from torch.utils.data import Dataset, DataLoader
from appfl.privacy import laplace_mechanism_output_perturb
from appfl.algorithm.trainer.base_trainer import BaseTrainer

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
        model: Optional[Module]=None,
        loss_fn: Optional[Module]=None,
        metric: Optional[Any]=None,
        train_dataset: Optional[Dataset]=None,
        val_dataset: Optional[Dataset]=None,
        train_configs: DictConfig = DictConfig({}),
        logger: Optional[Any]=None,
        **kwargs
    ):
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            metric=metric,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_configs=train_configs,
            logger=logger,
            **kwargs
        )
        if not hasattr(self.train_configs, "device"):
            self.train_configs.device = "cpu"
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.train_configs.get("train_batch_size", 32),
            shuffle=self.train_configs.get("train_data_shuffle", True),
            num_workers=self.train_configs.get("num_workers", 0),
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.train_configs.get("val_batch_size", 32),
            shuffle=self.train_configs.get("val_data_shuffle", False),
            num_workers=self.train_configs.get("num_workers", 0),
        ) if self.val_dataset is not None else None
        self._sanity_check()
        
    def train(self):
        """
        Train the model for a certain number of local epochs or steps and store the mode state
        (probably with perturbation for differential privacy) in `self.model_state`.
        """
        # Store the previous model state for gradient computation
        send_gradient = self.train_configs.get("send_gradient", False)
        if send_gradient:
            self.model_prev = copy.deepcopy(self.model.state_dict())

        self.model.to(self.train_configs.device)

        do_validation = self.train_configs.get("do_validation", False) and self.val_dataloader is not None
        do_pre_validation = self.train_configs.get("do_pre_validation", False) and do_validation
        
        # Set up logging title
        if self.round == 0:
            title = (
                ["Round", "Time", "Train Loss", "Train Accuracy"] 
                if not do_validation
                else (
                    ["Round", "Pre Val?", "Time", "Train Loss", "Train Accuracy", "Val Loss", "Val Accuracy"] 
                    if do_pre_validation 
                    else ["Round", "Time", "Train Loss", "Train Accuracy", "Val Loss", "Val Accuracy"]
                )
            )
            if self.train_configs.mode == "epoch":
                title.insert(1, "Epoch")
            self.logger.log_title(title)

        if do_pre_validation:
            val_loss, val_accuracy = self._validate()
            content = [self.round, "Y", " ", " ", " ", val_loss, val_accuracy]  
            if self.train_configs.mode == "epoch":
                content.insert(1, 0)
            self.logger.log_content(content)
        
        # Start training
        optim_module = importlib.import_module("torch.optim")
        assert hasattr(optim_module, self.train_configs.optim), f"Optimizer {self.train_configs.optim} not found in torch.optim"
        optimizer = getattr(optim_module, self.train_configs.optim)(self.model.parameters(), **self.train_configs.optim_args)
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
                target_true, target_pred = np.concatenate(target_true), np.concatenate(target_pred)
                train_accuracy = float(self.metric(target_true, target_pred))
                if do_validation:
                    val_loss, val_accuracy = self._validate()
                per_epoch_time = time.time() - start_time
                self.logger.log_content(
                    [self.round, epoch, per_epoch_time, train_loss, train_accuracy] 
                    if not do_validation
                    else (
                        [self.round, epoch, per_epoch_time, train_loss, train_accuracy, val_loss, val_accuracy] 
                        if not do_pre_validation 
                        else 
                        [self.round, epoch, 'N', per_epoch_time, train_loss, train_accuracy, val_loss, val_accuracy]
                    )
                )
        else:
            start_time = time.time()
            data_iter = iter(self.train_dataloader)
            train_loss, target_true, target_pred = 0, [], []
            for _ in range(self.train_configs.num_local_steps):
                try:
                    data, target = next(data_iter)
                except:
                    data_iter = iter(self.train_dataloader)
                    data, target = next(data_iter)
                loss, pred, label = self._train_batch(optimizer, data, target)
                train_loss += loss
                target_true.append(label)
                target_pred.append(pred)
            train_loss /= len(self.train_dataloader)
            target_true, target_pred = np.concatenate(target_true), np.concatenate(target_pred)
            train_accuracy = float(self.metric(target_true, target_pred))
            if do_validation:
                val_loss, val_accuracy = self._validate()
            per_step_time = time.time() - start_time
            self.logger.log_content(
                [self.round, per_step_time, train_loss, train_accuracy] 
                if not do_validation
                else (
                    [self.round, per_step_time, train_loss, train_accuracy, val_loss, val_accuracy]
                    if not do_pre_validation 
                    else 
                    [self.round, 'N', per_step_time, train_loss, train_accuracy, val_loss, val_accuracy]
                )
            )

        self.round += 1

        # Differential privacy
        if self.train_configs.get("use_dp", False):
            assert hasattr(self.train_configs, "clip_value"), "Gradient clipping value must be specified"
            assert hasattr(self.train_configs, "epsilon"), "Privacy budget (epsilon) must be specified"
            sensitivity = 2.0 * self.train_configs.clip_value * self.train_configs.optim_args.lr
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

    def get_parameters(self) -> Dict:
        if not hasattr(self, "model_state"):
            self.model_state = copy.deepcopy(self.model.state_dict())
        return self.model_state

    def _sanity_check(self):
        """
        Check if the configurations are valid.
        """
        assert hasattr(self.train_configs, "mode"), "Training mode must be specified"
        assert self.train_configs.mode in ["epoch", "step"], "Training mode must be either 'epoch' or 'step'"
        if self.train_configs.mode == "epoch":
            assert hasattr(self.train_configs, "num_local_epochs"), "Number of local epochs must be specified"
        else:
            assert hasattr(self.train_configs, "num_local_steps"), "Number of local steps must be specified"

    def _validate(self) -> Tuple[float, float]:
        """
        Validate the model
        :return: loss, accuracy
        """
        device = self.train_configs.device
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
        val_accuracy = float(self.metric(np.concatenate(target_true), np.concatenate(target_pred)))
        self.model.train()
        return val_loss, val_accuracy

    def _train_batch(self, optimizer: torch.optim.Optimizer, data, target) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Train the model for one batch of data
        :param optimizer: torch optimizer
        :param data: input data
        :param target: target label
        :return: loss, prediction, label
        """
        device = self.train_configs.device
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = self.model(data)
        loss = self.loss_fn(output, target)
        loss.backward()
        if getattr(self.train_configs, "clip_grad", False) or getattr(self.train_configs, "use_dp", False):
            assert hasattr(self.train_configs, "clip_value"), "Gradient clipping value must be specified"
            assert hasattr(self.train_configs, "clip_norm"), "Gradient clipping norm must be specified"
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
        if not hasattr(self, 'named_parameters'):
            self.named_parameters = set()
            for name, _ in self.model.named_parameters():
                self.named_parameters.add(name)
        for name in self.model_state:
            if name in self.named_parameters:
                self.model_state[name] = self.model_prev[name].cpu() - self.model_state[name]