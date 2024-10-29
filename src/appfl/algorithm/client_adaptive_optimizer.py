import os
import copy
import time
import torch
import numpy as np
from torch.optim import *
from .fl_base import BaseClient
from collections import OrderedDict

class ClientAdaptOptim(BaseClient):
    """This client optimizer performs updates for a certain number of epochs in each training round."""
    def __init__(self, id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader, metric, **kwargs):
        super(ClientAdaptOptim, self).__init__(id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader, metric)
        self.__dict__.update(kwargs)
        
        if hasattr(self, 'outfile') and self.outfile:
            self.client_log_title()

    def client_log_title(self):
        # 扩展标题以包括 LearningRate 和 GradNorm
        title = "%10s %10s %10s %10s %10s %10s %10s %10s %10s %10s\n" % (
            "Round",
            "LocalEpoch",
            "PerIter[s]",
            "TrainLoss",
            "TrainAccu",
            "TestLoss",
            "TestAccu",
            "LearningRate",
            "GradNorm",
            "ValueCheck"
        )
        self.outfile.write(title)
        self.outfile.flush()

    def client_log_content(
        self, t, per_iter_time, train_loss, train_accuracy, test_loss, test_accuracy, learning_rate, grad_norm, value_check
    ):
        # 扩展内容格式，以记录 LearningRate 和 GradNorm
        contents = "%10s %10s %10.2f %10.4f %10.4f %10.4f %10.4f %10.6f %10.4f %10.4f\n" % (
            self.round,
            t,
            per_iter_time,
            train_loss,
            train_accuracy,
            test_loss,
            test_accuracy,
            learning_rate,
            grad_norm,
            value_check
        )
        self.outfile.write(contents)
        self.outfile.flush()

    def update(self, global_model, learning_rate):
        #print(f"Client {self.id} received learning rate: {learning_rate}")
        """
        Perform local updates using the provided global model and learning rate.
        Args:
            global_model: The global model parameters received from the server.
            learning_rate: The learning rate provided by the server.
        """
        # Load the global model weights
        self.model.load_state_dict(global_model)
        # learning rate from server
        self.optim_args["lr"] = learning_rate
        self.model.to(self.cfg.device)
        optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)
        #print(f"Client {self.id} optimizer learning rate: {optimizer.param_groups[0]['lr']}")  # Print optimizer LR for verification
        initial_model_state = copy.deepcopy(self.model.state_dict())  # Save initial state for gradient estimate

        # Compute the initial loss
        initial_loss = 0
        with torch.no_grad():
            for data, target in self.dataloader:
                data = data.to(self.cfg.device)
                target = target.to(self.cfg.device)
                output = self.model(data)
                loss = self.loss_fn(output, target)
                initial_loss += loss.item()
        initial_loss /= len(self.dataloader)

        # Calculate initial gradient norm
        gradient_norm = 0
        for data, target in self.dataloader:
            data = data.to(self.cfg.device)
            target = target.to(self.cfg.device)
            optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            gradient_norm += torch.norm(
                torch.cat([param.grad.view(-1) for param in self.model.parameters() if param.grad is not None])
            ).item()
        gradient_norm /= len(self.dataloader)  # Compute average gradient norm

        # Initial evaluation with value_check calculation
        function_value_difference = initial_loss
        value_check = function_value_difference + learning_rate * (gradient_norm ** 2)

        # Log initial values
        self.client_log_content(0, 0, initial_loss, 0, 0, 0, learning_rate, gradient_norm, value_check)

        # Local training
        for t in range(self.num_local_epochs):
            start_time = time.time()
            train_loss, target_true, target_pred = 0, [], []
            for data, target in self.dataloader:
                data = data.to(self.cfg.device)
                target = target.to(self.cfg.device)
                optimizer.zero_grad()
                output = self.model(data)  # Forward pass
                loss = self.loss_fn(output, target)  # Compute loss
                loss.backward()

                # 计算梯度范数
                gradient_norm = torch.norm(
                    torch.cat([param.grad.view(-1) for param in self.model.parameters() if param.grad is not None])
                ).item()

                optimizer.step()

                target_true.append(target.detach().cpu().numpy())
                target_pred.append(output.detach().cpu().numpy())
                train_loss += loss.item()

                # Compute the real gradient norm for this batch
                # total_grad_norm = 0.0
                # batch_grad_norm = 0.0
                # for param in self.model.parameters():
                #     if param.grad is not None:
                #         batch_grad_norm += param.grad.norm(2).item() ** 2  # L2 norm squared for each parameter
                # batch_grad_norm = batch_grad_norm ** 0.5  # Take the square root to get the final gradient norm
                
                # total_grad_norm += batch_grad_norm
                # print("the real gradient values: ",total_grad_norm )

                if self.clip_grad or self.use_dp:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.clip_value,
                        norm_type=self.clip_norm,
                    )

            train_loss /= len(self.dataloader)
            # print(f"Client {self.id} Epoch {t} Average Training Loss: {train_loss}")  # Print average loss for each epoch
            target_true, target_pred = np.concatenate(target_true), np.concatenate(target_pred)
            train_accuracy = float(self.metric(target_true, target_pred))

            # Validation
            if self.cfg.validation and self.test_dataloader is not None:
                test_loss, test_accuracy = super(ClientAdaptOptim, self).client_validation()
            else:
                test_loss, test_accuracy = 0, 0

            per_iter_time = time.time() - start_time

            # Calculate value_check based on current epoch's data
            function_value_difference = train_loss - initial_loss
            value_check = function_value_difference + learning_rate * (gradient_norm ** 2)

            # Log values for each epoch
            self.client_log_content(t + 1, per_iter_time, train_loss, train_accuracy, test_loss, test_accuracy, learning_rate, gradient_norm, value_check)

            # Save model.state_dict()
            if self.cfg.save_model_state_dict:
                path = self.cfg.output_dirname + f"/client_{self.id}"
                if not os.path.exists(path):
                    os.makedirs(path, exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(path, f"{self.round}_{t}.pt"))

        self.round += 1

        # Compute gradient estimate
        self.grad_estimate = OrderedDict()
        for name, param in self.model.named_parameters():
            # gradient for each parameter
            # self.grad_estimate[name] = (initial_model_state[name] - param.data) / learning_rate
            self.grad_estimate[name] = param.grad

        # Final loss computation
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

        # client selection criteria check: 
        value_check = self.function_value_difference + learning_rate* (gradient_norm**2)

        # Differential Privacy
        self.primal_state = copy.deepcopy(self.model.state_dict())
        if self.use_dp:
            sensitivity = 2.0 * self.clip_value * self.optim_args['lr']
            scale_value = sensitivity / self.epsilon
            super(ClientAdaptOptim, self).laplace_mechanism_output_perturb(scale_value)

        # Move model parameters to CPU for communication
        # if self.cfg.device == "cuda":
        #     for k in self.primal_state:
        #         self.primal_state[k] = self.primal_state[k].cpu()

        return {
            "primal_state": self.primal_state,
            "grad_estimate": self.grad_estimate,
            "function_value_difference": self.function_value_difference
        }
