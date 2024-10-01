import os
import copy
import time
import torch
import numpy as np
from torch.optim import *
from .fl_base import BaseClient
from collections import OrderedDict

class ClientAdaptOptim(BaseClient):
    """This client optimizer which perform updates for certain number of epochs in each training round."""
    def __init__(
        self, id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader, metric, **kwargs
    ):      
        super(ClientAdaptOptim, self).__init__(id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader, metric)
        self.__dict__.update(kwargs)
        super(ClientAdaptOptim, self).client_log_title()

        self.clip_grad = cfg.get('clip_grad',False)
        self.clip_value = cfg.get('clip_value',1.0)
        self.clip_norm = self.cfg.get('clip_norm', 2)

    def clip_gradient_estimate(self, grad_estimate, max_norm, norm_type=2):
        """
        Clips the gradient estimate to ensure its norm is within the specified limit.

        Args:
            grad_estimate: The gradient estimate (difference in weights divided by lr).
            max_norm: The maximum allowable norm for the gradient estimate.
            norm_type: The type of norm (default is L2 norm).
        """
        total_norm = 0.0
        # Calculate the total norm of the gradient estimate
        for name, grad in grad_estimate.items():
            param_norm = grad.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        # if norm type =2, calculate L2 norm, if norm type =1, takes L1 norm.
        total_norm = total_norm ** (1. / norm_type)

        # If the total norm exceeds the max norm, scale the gradients
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for name, grad in grad_estimate.items():
                grad_estimate[name].mul_(clip_coef)  # Scale the gradient estimate
        return grad_estimate

    def update(self, lr):
        
        self.optim_args["lr"] = lr        
        
        # Load the global model weights
        self.model.to(self.cfg.device)
        optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)
        initial_model_state = copy.deepcopy(self.model.state_dict())  # Save initial state for gradient estimate

        initial_loss = 0
        with torch.no_grad():
            for data, target in self.dataloader:
                data = data.to(self.cfg.device)
                target = target.to(self.cfg.device)
                output = self.model(data)
                loss = self.loss_fn(output, target)
                initial_loss += loss.item()
        initial_loss /= len(self.dataloader)

        # Initial evaluation
        if self.cfg.validation and self.test_dataloader is not None:
            start_time = time.time()
            test_loss, test_accuracy = super(ClientAdaptOptim, self).client_validation()
            per_iter_time = time.time() - start_time
            super(ClientAdaptOptim, self).client_log_content(0, per_iter_time, 0, 0, test_loss, test_accuracy)

        # Local training
        for t in range(self.num_local_epochs):
            start_time = time.time()
            train_loss, target_true, target_pred = 0, [], []
            for data, target in self.dataloader:
                data = data.to(self.cfg.device)
                target = target.to(self.cfg.device)
                optimizer.zero_grad()
                output = self.model(data)  # Forward pass to get the model's output
                loss = self.loss_fn(output, target)  # Calculate the loss using the output (logits)
                loss.backward()            
                optimizer.step()

                # Log results
                target_true.append(target.detach().cpu().numpy())
                target_pred.append(output.detach().cpu().numpy())
                train_loss += loss.item()
                if self.clip_grad or self.use_dp:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.clip_value,
                        norm_type=self.clip_norm,
                    )


            train_loss /= len(self.dataloader)
            target_true, target_pred = np.concatenate(target_true), np.concatenate(target_pred)
            train_accuracy = float(self.metric(target_true, target_pred))


            # Validation
            if self.cfg.validation and self.test_dataloader is not None:
                test_loss, test_accuracy = super(ClientAdaptOptim, self).client_validation()
            else:
                test_loss, test_accuracy = 0, 0

            per_iter_time = time.time() - start_time
            super(ClientAdaptOptim, self).client_log_content(t+1, per_iter_time, train_loss, train_accuracy, test_loss, test_accuracy)

            # Save model.state_dict()
            if self.cfg.save_model_state_dict:
                path = self.cfg.output_dirname + f"/client_{self.id}"
                if not os.path.exists(path):
                    os.makedirs(path, exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(path, f"{self.round}_{t}.pt"))

        self.round += 1

        # Compute gradient estimate using stochastic oracle
        self.grad_estimate = OrderedDict()
        for name, param in self.model.named_parameters():
            self.grad_estimate[name] = (initial_model_state[name] - param.data) / self.optim_args['lr']
 

        # self.clip_gradient_estimate(self.grad_estimate, max_norm=self.clip_value, norm_type=self.clip_norm)

       # loss after training
        final_loss = 0
        with torch.no_grad():
            for data, target in self.dataloader:
                data = data.to(self.cfg.device)
                target = target.to(self.cfg.device)
                output = self.model(data)
                loss = self.loss_fn(output, target)
                final_loss += loss.item()
        final_loss /= len(self.dataloader)

        self.function_value_difference = final_loss - initial_loss
 

        # Differential Privacy
        self.primal_state = copy.deepcopy(self.model.state_dict())
        if self.use_dp:
            sensitivity = 2.0 * self.clip_value * self.optim_args['lr']
            scale_value = sensitivity / self.epsilon
            super(ClientAdaptOptim, self).laplace_mechanism_output_perturb(scale_value)

        # Move the model parameters to CPU for communication (if necessary)
        if self.cfg.device == "cuda":
            for k in self.primal_state:
                self.primal_state[k] = self.primal_state[k].cpu()
                                
        return {
            "primal_state": self.primal_state,
            "grad_estimate": self.grad_estimate,  # Corrected this line to use grad_estimate
            "function_value_difference": self.function_value_difference
        }
