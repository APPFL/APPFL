from .fl_base import BaseServer, BaseClient
from collections import OrderedDict
from torch.optim import *
from collections import OrderedDict
import torch
import copy
import logging

from .server_federated import FedServer
from ..misc import *

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
