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
            # Move gradients and function value differences to GPU
            gradients[client_id] = {k: v.to(self.device) for k, v in state['gradient_state'].items()}
            func_val_diffs[client_id] = torch.tensor(state['function_value_difference'], device=self.device)

        # Accumulated change in objective function across all clients
        total_func_val_diff = sum(func_val_diffs.values())

        # Select clients whose updates meet the condition
        # selected_clients = [
        #     client_id for client_id in range(self.num_clients)
        #     if func_val_diffs[client_id] <= -self.server_lr * torch.norm(
        #         torch.cat([gradients[client_id][k].view(-1) for k in gradients[client_id]])
        #     ).item() ** 2
        # ]
        selected_clients = []
        for client_id in range(self.num_clients):
            # Compute the norm of the client's gradient
            gradient_norm = torch.norm(torch.cat([gradients[client_id][k].view(-1) for k in gradients[client_id]])).item()

            # Apply the selection condition
            if total_func_val_diff <= -self.server_lr * gradient_norm ** 2:
                selected_clients.append(client_id)



        # Initialize global state for aggregation on GPU
        global_state = copy.deepcopy(self.model.state_dict())
        global_state = {k: v.to(self.device) for k, v in global_state.items()}

        if selected_clients:
            # Update the global model using only the selected clients
            for key in global_state.keys():
                global_state[key] = torch.zeros_like(global_state[key], device=self.device)
                for client_id in selected_clients:

                    # Move the client's local state to the server's device before aggregation
                    local_param = local_states[client_id]['primal_state'][key].to(self.device)
                    global_state[key] = global_state[key].float()
                    global_state[key] += local_param * self.weights[client_id]

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

    def logging_summary(self, cfg, logger):
        super(FedServer, self).log_summary(cfg, logger)
        logger.info("client_learning_rate = %s " % (cfg.fed.args.optim_args.lr))
        if cfg.summary_file != "":
            with open(cfg.summary_file, "a") as f:
                f.write(
                    cfg.logginginfo.DataSet_name
                    + " FedAvg ClientLR "
                    + str(cfg.fed.args.optim_args.lr)
                    + " TestAccuracy "
                    + str(cfg.logginginfo.accuracy)
                    + " BestAccuracy "
                    + str(cfg.logginginfo.BestAccuracy)
                    + " Time "
                    + str(round(cfg.logginginfo.Elapsed_time, 2))
                    + "\n"
                )