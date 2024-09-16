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

    def update(self, local_states):
        """
        Update the global model by selecting clients based on their contributions and adjusting learning rates.

        Args:
            local_states (list): A list of dictionaries, where each dictionary contains:
                - 'primal_state': Local model parameters from the client.
                - 'gradient_state': Local gradients from the client.
                - 'function_value_difference': The change in the objective function value from the client.
        """
        # Initialize storage for gradients and function value differences
        gradients = OrderedDict()
        func_val_diffs = OrderedDict()

        # Gather gradients and function value differences from clients
        for client_id, state in enumerate(local_states):
            gradients[client_id] = state['gradient_state']
            func_val_diffs[client_id] = state['function_value_difference']

        # Accumulated change in objective function across all clients
        total_func_val_diff = sum(func_val_diffs.values())

        # Select clients whose updates meet the condition
        selected_clients = [
            client_id for client_id in range(self.num_clients)
            if func_val_diffs[client_id] <= -self.server_lr * torch.norm(torch.cat([gradients[client_id][k].view(-1) for k in gradients[client_id]])).item() ** 2
        ]

        # Initialize global state for aggregation
        global_state = copy.deepcopy(self.model.state_dict())
        
        if selected_clients:
            # Update the global model using only the selected clients
            for key in self.model.state_dict().keys():
                if key in global_state:  # Ensure key exists in global model
                    global_state[key] = torch.zeros_like(self.model.state_dict()[key], device=self.device)
                    for client_id in selected_clients:
                        global_state[key] += local_states[client_id]['primal_state'][key] * self.weights[client_id]
                    # Normalize by the number of selected clients
                    global_state[key] /= len(selected_clients)
        
            # Load the updated global state into the model
            self.model.load_state_dict(global_state)

        # Update learning rates for clients
        for client_id in range(self.num_clients):
            if client_id in selected_clients:
                self.weights[client_id] *= self.gamma  # Increase learning rate for selected clients
            else:
                self.weights[client_id] /= self.gamma  # Decrease learning rate for non-selected clients

    def logging_iteration(self, cfg, logger, t):
        if t == 0:
            title = super(AdaptiveFLServer, self).log_title()
            logger.info(title)
        contents = super(AdaptiveFLServer, self).log_contents(cfg, t)
        logger.info(contents)

