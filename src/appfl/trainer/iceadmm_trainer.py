import copy
import time
import torch
import importlib
import numpy as np
import torch.nn as nn
from omegaconf import DictConfig
from collections import OrderedDict
from torch.utils.data import DataLoader
from typing import Any, Optional, Tuple
from appfl.trainer.base_trainer import BaseTrainer
from appfl.privacy import laplace_mechanism_output_perturb

class ICEADMMTrainer(BaseTrainer):
    """
    ICEADMM Trainer:
        Local trainer for the ICEADMM algorithm.
        This trainer must be used with the ICEADMMAggregator.
    """
    def __init__(
        self,
        model: Optional[nn.Module]=None,
        loss_fn: Optional[nn.Module]=None,
        metric: Optional[Any]=None,
        train_dataset: Optional[DataLoader]=None,
        val_dataset: Optional[DataLoader]=None,
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

        self.penalty = self.train_configs.get("init_penalty", 500.0)
        self.proximity = self.train_configs.get("init_proximity", 0)
        self.is_first_iter = True

        # At initial, (1) primal_states = global_state, (2) dual_states = 0
        self.primal_states = OrderedDict()
        self.dual_states = OrderedDict()
        self.primal_states_curr = OrderedDict()
        self.primal_states_prev = OrderedDict()
        self.named_parameters = set()

        for name, param in self.model.named_parameters():
            self.named_parameters.add(name)
            self.primal_states[name] = param.data
            self.dual_states[name] = torch.zeros_like(param.data)

    def train(self):
        self._sanity_check()
        self.model.train()
        self.model.to(self.train_configs.device)
        do_validation = self.train_configs.get("do_validation", False) and self.val_dataloader is not None
        do_pre_validation = self.train_configs.get("do_pre_validation", False) and do_validation
        
        """Set up logging title"""
        if self.round == 0:
            title = (
                ["Round", "Pre Val?" "Time", "Train Loss", "Train Accuracy"] 
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
            self.logger.log_content([self.round, "Y", " ", " ", " ", val_loss, val_accuracy])

        optim_module = importlib.import_module("torch.optim")
        assert hasattr(optim_module, self.train_configs.optim), f"Optimizer {self.train_configs.optim} not found in torch.optim"
        optimizer = getattr(optim_module, self.train_configs.optim)(self.model.parameters(), **self.train_configs.optim_args)

        """ Inputs for the local model update """
        global_state = copy.deepcopy(self.model.state_dict())

        """Adaptive Penalty (Residual Balancing)"""
        if not hasattr(self.train_configs, "residual_balancing"):
            self.train_configs.residual_balancing = DictConfig({})
        if getattr(self.train_configs.residual_balancing, "res_on", False):
            prim_res = self._primal_residual_at_client(global_state)
            dual_res = self._dual_residual_at_client()
            self._residual_balancing(prim_res, dual_res)

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


        for name, param in self.model.named_parameters():
            param.data = self.primal_states[name].to(self.train_configs.device)
        if self.train_configs.get("use_dp", False):
            sensitivity = 2.0 * self.train_configs.clip_value / self.penalty
            self._model_state = laplace_mechanism_output_perturb(
                self.model,
                sensitivity,
                self.train_configs.epsilon
            )
        else:
            self._model_state = copy.deepcopy(self.model.state_dict())

        """Move to CPU for communication"""
        if self.train_configs.get("device", "cpu") == "cuda":
            for k in self._model_state:
                self._model_state[k] = self._model_state[k].cpu()
            for name in self.named_parameters:
                self.dual_states[name] = self.dual_states[name].cpu()

        self.model_state = OrderedDict()
        self.model_state["primal"] = self._model_state
        self.model_state["dual"] = self.dual_states
        self.model_state["penalty"] = self.penalty

    def get_parameters(self) -> OrderedDict:
        hasattr(self, "model_state"), "Please make sure the model has been trained before getting its parameters"
        return self.model_state

    def set_weight(self, weight=1.0):
        """Set the weight of the client"""
        self.weight = weight

    def _sanity_check(self):
        """
        Check if the necessary configurations are provided.
        """
        assert hasattr(self.train_configs, "mode"), "Training mode must be specified"
        assert self.train_configs.mode in ["epoch", "step"], "Training mode must be either 'epoch' or 'step'"
        if self.train_configs.mode == "epoch":
            assert hasattr(self.train_configs, "num_local_epochs"), "Number of local epochs must be specified"
        else:
            assert hasattr(self.train_configs, "num_local_steps"), "Number of local steps must be specified"
        assert hasattr(self, "weight"), "You must set the weight of the client before training. Use `set_weight` method."
        if getattr(self.train_configs, "clip_grad", False) or getattr(self.train_configs, "use_dp", False):
            assert hasattr(self.train_configs, "clip_value"), "Gradient clipping value must be specified"
            assert hasattr(self.train_configs, "clip_norm"), "Gradient clipping norm must be specified"
            if getattr(self.train_configs, "use_dp", False):
                assert hasattr(self.train_configs, "epsilon"), "Privacy budget (epsilon) must be specified"

    def _primal_residual_at_client(self, global_state) -> float:
        """
        Calculate primal residual.
        :param global_state: global state - input for the local model update
        :return: primal residual
        """
        primal_res = 0
        for name in self.named_parameters:
            primal_res += torch.sum(
                torch.square(
                    global_state[name].to(self.train_configs.device)
                    - self.primal_states[name].to(self.train_configs.device)
                )
            )
        primal_res = torch.sqrt(primal_res).item()
        return primal_res
    
    def _dual_residual_at_client(self) -> float:
        """
        Calculate dual residual.
        :return: dual residual
        """
        dual_res = 0
        if self.is_first_iter:
            self.primal_states_curr = self.primal_states
            self.is_first_iter = False
        else:
            self.primal_states_prev = self.primal_states_curr
            self.primal_states_curr = self.primal_states
            for name in self.named_parameters:
                res = self.penalty * (
                    self.primal_states_prev[name] - self.primal_states_curr[name]
                )
                dual_res += torch.sum(torch.square(res))
            dual_res = torch.sqrt(dual_res).item()
        return dual_res

    def _residual_balancing(self, prim_res, dual_res):
        if prim_res > self.train_configs.residual_balancing.mu * dual_res:
            self.penalty = self.penalty * self.train_configs.residual_balancing.tau
        if dual_res > self.train_configs.residual_balancing.mu * prim_res:
            self.penalty = self.penalty / self.train_configs.residual_balancing.tau

    def _train_batch(self, optimizer, data, target, global_state):
        """
        Train the model for one batch of data
        :param optimizer: torch optimizer
        :param data: input data
        :param target: target label
        :param global_state: global model state
        :return: loss, prediction, label
        """
        """Load primal states to the model"""
        for name, param in self.model.named_parameters():
            param.data = self.primal_states[name].to(self.train_configs.device)
        
        """Adaptive Penalty (Residual Balancing)"""
        if (
            getattr(self.train_configs.residual_balancing, "res_on", False) and 
            getattr(self.train_configs.residual_balancing, "res_on_every_update", False)
        ):
            prim_res = self._primal_residual_at_client(global_state)
            dual_res = self._dual_residual_at_client()
            self._residual_balancing(prim_res, dual_res)

        """Train the model"""
        data, target = data.to(self.train_configs.device), target.to(self.train_configs.device)
        if getattr(self.train_configs, "accum_grad", False) == False:
            optimizer.zero_grad()
        output = self.model(data)            
        loss = self.loss_fn(output, target)
        loss.backward()
        optimizer.step()

        """Gradient Clipping"""
        if getattr(self.train_configs, "clip_grad", False) or getattr(self.train_configs, "use_dp", False):
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.train_configs.clip_value,
                norm_type=self.train_configs.clip_norm,
            )
    
        """Update primal and dual states"""
        coefficient = 1
        if getattr(self.train_configs, "coeff_grad", False):
            coefficient = self.weight * len(target) / len(self.train_dataloader.dataset)
        self._iceadmm_step(coefficient, global_state)

        return loss.item(), output.detach().cpu().numpy(), target.detach().cpu().numpy()

    def _iceadmm_step(self, coefficient, global_state):
        """
        Update primal and dual states
        """
        for name, param in self.model.named_parameters():
            self.primal_states[name] = self.primal_states[name].to(self.train_configs.device)
            self.dual_states[name] = self.dual_states[name].to(self.train_configs.device)
            global_state[name] = global_state[name].to(self.train_configs.device)

            grad = param.grad * coefficient
            """Update primal"""
            self.primal_states[name] = self.primal_states[name] - (
                self.penalty * (self.primal_states[name] - global_state[name])
                + grad
                + self.dual_states[name]
            ) / (self.weight * self.proximity + self.penalty)
            """Update dual"""
            self.dual_states[name] = self.dual_states[name] + self.penalty * (
                self.primal_states[name] - global_state[name]
            )

    def _validate(self) -> Tuple[float, float]:
        """
        Validate the model
        :return: loss, accuracy
        """
        device = self.train_configs.get("device", "cpu")
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