from .fl_base import BaseServer, BaseClient
from collections import OrderedDict
import torch

class adaptive_fl_Server(BaseServer):
    def __init__(self, weights, model, num_clients, device, server_lr=1.0, gamma=1.1, **kwargs):
        """
        Initialize the AdaptiveFLServer class.

        Args:
            weights (OrderedDict): Aggregation weights for each client.
            model (nn.Module): The global model to be trained.
            num_clients (int): The number of clients.
            device (str): The device for computation (e.g., 'cpu' or 'cuda').
            server_lr (float): Learning rate for the server's update.
            gamma (float): Multiplicative factor for adapting learning rates.
            **kwargs: Additional keyword arguments.
        """
        super(adaptive_fl_Server, self).__init__(weights, model, num_clients, device)
        self.server_lr = server_lr
        self.gamma = gamma
        self.__dict__.update(kwargs)
        # Any additional initialization

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


class adaptive_fl_Client(BaseClient):
    def __init__(self, id, weight, model, dataloader, device, initial_lr=0.1, gamma=1.1, **kwargs):
        """
        Initialize the AdaptiveFLClient class.

        Args:
            id (int): Unique ID for each client.
            weight (Dict): Aggregation weight for the client.
            model (nn.Module): The local model to be trained.
            dataloader (DataLoader): The client's data loader.
            device (str): The device for computation (e.g., 'cpu' or 'cuda').
            initial_lr (float): Initial learning rate for the client.
            gamma (float): Multiplicative factor for adapting learning rates.
            **kwargs: Additional keyword arguments.
        """
        super(adaptive_fl_Client, self).__init__(id, weight, model, dataloader, device)
        self.lr = initial_lr
        self.gamma = gamma
        self.__dict__.update(kwargs)
        # Any additional initialization

    def stochastic_oracle(self, global_model):
        """
        Compute the gradient and function value difference using stochastic oracles.
        Args:
            global_model (nn.Module): The global model sent by the server.

        Returns:
            gradient (torch.Tensor): Estimated gradient.
            func_val_diff (float): Estimated function value difference.
        """
        # Initialize model with global weights
        self.model.load_state_dict(global_model)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        # Mini-batch sampling
        data, target = next(iter(self.dataloader))
        data, target = data.to(self.device), target.to(self.device)

        # Forward pass
        optimizer.zero_grad()
        output = self.model(data)
        loss = self.loss_fn(output, target)

        # Backward pass and gradient computation
        loss.backward()
        optimizer.step()

        # Compute gradient and function value difference
        gradient = self._compute_gradient()
        func_val_diff = self._compute_func_val_diff(global_model, data, target)
        
        return gradient, func_val_diff

    def update(self):
        """
        Perform local training using stochastic oracles and return the resulting model parameters, gradient, and function value difference.
        """
        gradient, func_val_diff = self.stochastic_oracle(self.model.state_dict())
        return self.model.state_dict(), gradient, func_val_diff





