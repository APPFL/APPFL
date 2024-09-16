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


    def update(self):
        # Load the global model weights
        self.model.to(self.cfg.device)
        optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)
        
        initial_loss = 0
        with torch.no_grad():
            for data, target in self.dataloader:
                data = data.to(self.cfg.device)
                target = target.to(self.cfg.device)
                output = self.model(data)
                loss = self.loss_fn(output, target)
                initial_loss += loss.item()
        initial_loss /= len(self.dataloader)

        # gradients
        self.gradient_state = OrderedDict()
        for name, param in self.model.named_parameters():
            self.gradient_state[name] = torch.zeros_like(param, device='cpu')

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

                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        self.gradient_state[name] += param.grad.clone().detach().cpu()
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
            # averaging gradients over number of mini-batches
            num_batches = len(self.dataloader) * self.num_local_epochs
            for name in self.gradient_state:
                self.gradient_state[name] /= num_batches


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
            "gradient_state": self.gradient_state,
            "function_value_difference": self.function_value_difference
        }
