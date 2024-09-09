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

   """  def update(self):
         
        self.model.to(self.cfg.device)
        optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)

        initial_model_state = copy.deepcopy(self.model.state_dict())  # Save initial state for gradient estimate

        # Initial evaluation
        if self.cfg.validation == True and self.test_dataloader != None:
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
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()
                optimizer.step()
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
            if self.cfg.validation == True and self.test_dataloader != None:
                test_loss, test_accuracy = super(ClientAdaptOptim, self).client_validation()
            else:
                test_loss, test_accuracy = 0, 0
            per_iter_time = time.time() - start_time
            super(ClientAdaptOptim, self).client_log_content(t+1, per_iter_time, train_loss, train_accuracy, 0, 0)

            # Save model.state_dict()
            if self.cfg.save_model_state_dict == True:
                path = self.cfg.output_dirname + "/client_%s" % (self.id)
                if not os.path.exists(path):
                    os.makedirs(path, exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(path, "%s_%s.pt" % (self.round, t)))
 
        self.round += 1

        # Compute gradient estimate using stochastic oracle
        grad_estimate = OrderedDict()
        for name, param in self.model.named_parameters():
            grad_estimate[name] = (initial_model_state[name] - param.data) / self.optim_args['lr']

        # Compute function value difference using stochastic oracle
        func_value_diff = OrderedDict()
        for name, param in self.model.named_parameters():
            func_value_diff[name] = self.loss_fn(self.model, param) - self.loss_fn(self.model, initial_model_state[name])

        # Differential Privacy
        self.primal_state = copy.deepcopy(self.model.state_dict())

        if self.use_dp:
            sensitivity = 2.0 * self.clip_value * self.optim_args['lr']
            scale_value = sensitivity / self.epsilon
            super(ClientAdaptOptim, self).laplace_mechanism_output_perturb(scale_value)

        # Move the model parameter to CPU (if not) for communication
        if self.cfg.device == "cuda":            
            for k in self.primal_state:
                self.primal_state[k] = self.primal_state[k].cpu()

        return {
            "primal_state": self.primal_state,
            "grad_estimate": grad_estimate,
            "func_value_diff": func_value_diff
        }
 """
    def update(self):
    # Load the global model weights
    self.model.to(self.cfg.device)
    optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)
    
    # Save initial model state for gradient estimation
    initial_model_state = copy.deepcopy(self.model.state_dict())

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

            # Apply gradient clipping if necessary
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

    # Stochastic Oracle - Gradient Estimation and Function Value Difference
    # Perform a forward pass and compute the gradient estimate and function value difference

    # Mini-batch sampling (taking one batch)
    data, target = next(iter(self.dataloader))
    data, target = data.to(self.cfg.device), target.to(self.cfg.device)

    # Reset the model to the initial state for proper gradient estimation
    self.model.load_state_dict(initial_model_state)

    # Forward pass before the local update
    optimizer.zero_grad()
    initial_output = self.model(data)  # Get the model's output (logits)
    initial_loss = self.loss_fn(initial_output, target)  # Calculate loss using logits

    # Backward pass to compute gradients
    initial_loss.backward()
    optimizer.step()

    # Compute gradient estimate based on the stochastic oracle
    grad_estimate = OrderedDict()
    for name, param in self.model.named_parameters():
        grad_estimate[name] = param.grad.data.clone()
        
    print(grad_estimate)  #TEST
    
    # Forward pass after the local update to calculate the function value difference
    updated_output = self.model(data)  # Get the updated model's output (logits)
    updated_loss = self.loss_fn(updated_output, target)  # Calculate loss using logits

    # Compute the function value difference: (updated loss - initial loss)
    func_value_diff = updated_loss - initial_loss.item()

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
        "grad_estimate": grad_estimate,
        "func_value_diff": func_value_diff
    }


    