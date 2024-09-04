from .fl_base import BaseServer, BaseClient
from collections import OrderedDict
from torch.optim import *
import torch
import copy
import logging


class AdaptiveFLServer(BaseServer):
    def __init__(self, weights, model, loss_fn, num_clients, device, server_lr=1.0, gamma=1.1, **kwargs):
        """
        Initialize the AdaptiveFLServer class.

        Args:
            weights (OrderedDict): Aggregation weights for each client.
            model (nn.Module): The global model to be trained.
            loss_fn (nn.Module): Loss function used for training.
            num_clients (int): The number of clients.
            device (str): The device for computation (e.g., 'cpu' or 'cuda').
            server_lr (float): Learning rate for the server's update.
            gamma (float): Multiplicative factor for adapting learning rates.
            **kwargs: Additional keyword arguments.
        """
        super(AdaptiveFLServer, self).__init__(weights, model, loss_fn, num_clients, device)
        self.server_lr = server_lr
        self.gamma = gamma
        self.__dict__.update(kwargs)
        # Additional initialization if needed

    def update(self, local_states: OrderedDict, gradients: OrderedDict, func_val_diffs: OrderedDict):
        """
        Update the global model by averaging the gradients from selected clients.

        Args:
            local_states (OrderedDict): A dictionary containing the local model states from clients.
            gradients (OrderedDict): A dictionary containing the gradients from clients.
            func_val_diffs (OrderedDict): A dictionary containing the function value differences from clients.
        """
        selected_clients = []
        global_state = OrderedDict()

        # Accumulated change in objective function across all clients
        total_func_val_diff = sum(func_val_diffs.values())

        # Select clients whose updates meet the condition
        for client_id in range(self.num_clients):
            if total_func_val_diff <= -self.server_lr * torch.norm(gradients[client_id]) ** 2:
                selected_clients.append(client_id)

        # Update the global model using only the selected clients
        for key in self.model.state_dict().keys():
            global_state[key] = torch.zeros_like(self.model.state_dict()[key])
            for client_id in selected_clients:
                global_state[key] += local_states[client_id][key] * self.weights[client_id]

        if selected_clients:
            global_state = {k: v / len(selected_clients) for k, v in global_state.items()}
            self.model.load_state_dict(global_state)

        # Update learning rates for clients
        for client_id in range(self.num_clients):
            if client_id in selected_clients:
                self.weights[client_id] *= self.gamma  # Increase learning rate for selected clients
            else:
                self.weights[client_id] /= self.gamma  # Decrease learning rate for non-selected clients

        
        def logging_iteration(self, cfg, logger, t):
            if t == 0:
                title = super(FedServer, self).log_title()
                logger.info(title)
            contents = super(FedServer, self).log_contents(cfg, t)
            logger.info(contents)


class AdaptiveFLClient(BaseClient):
    def __init__(
        self, id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader=None, metric=None, initial_lr=0.1, gamma=1.1, **kwargs
    ):
        """
        Initialize the AdaptiveFLClient class.

        Args:
            id (int): Unique ID for each client.
            weight (Dict): Aggregation weight for the client.
            model (nn.Module): The local model to be trained.
            loss_fn (nn.Module): Loss function used for training.
            dataloader (DataLoader): The client's data loader.
            cfg (DictConfig): Configuration settings.
            outfile (str): Log file for output.
            test_dataloader (DataLoader, optional): Test data loader for validation.
            metric (callable, optional): Metric for performance evaluation.
            initial_lr (float): Initial learning rate for the client.
            gamma (float): Multiplicative factor for adapting learning rates.
            **kwargs: Additional keyword arguments.
        """
        super(AdaptiveFLClient, self).__init__(id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader, metric)
        self.lr = initial_lr
        self.gamma = gamma
        self.__dict__.update(kwargs)
        # Additional initialization if needed

    def stochastic_oracle(self, global_model):
        """
        Compute the gradient and function value difference using stochastic oracles.
        """
        # Load the global model weights
        self.model.load_state_dict(global_model)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr) 

        # Mini-batch sampling
        data, target = next(iter(self.dataloader))
        data, target = data.to(self.cfg.device), target.to(self.cfg.device)

        # Forward pass
        optimizer.zero_grad()
        output = self.model(data)
        loss = self.loss_fn(output, target)

        # Backward pass and gradient computation
        loss.backward()
        optimizer.step()

        # Compute the gradient estimate
        gradient = OrderedDict()
        for name, param in self.model.named_parameters():
            gradient[name] = param.grad.data.clone()

        # Compute the function value difference
        func_val_diff = OrderedDict()
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                func_val_diff[name] = self.loss_fn(self.model(data), target).item() - loss.item()

        return gradient, func_val_diff

    def update(self):
        """
        Perform local training using stochastic oracles and return the resulting model parameters, gradient, and function value difference.
        """
        # Use the model state as the initial point for the stochastic oracle
        gradient, func_val_diff = self.stochastic_oracle(self.model.state_dict())
        return self.model.state_dict(), gradient, func_val_diff


