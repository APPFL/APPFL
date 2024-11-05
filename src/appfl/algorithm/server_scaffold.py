import random
from collections import OrderedDict
import torch
import copy
from .server_federated import FedServer

class SCAFFOLDServer(FedServer):
    def __init__(self, weights, model, loss_fn, num_clients, device, server_lr=1e-1, sample_fraction=0.1, **kwargs):
        super(SCAFFOLDServer, self).__init__(weights, model, loss_fn, num_clients, device)
        self.server_lr = server_lr
        self.sample_fraction = sample_fraction  # Fraction of clients to sample each round
        self.global_control = {k: torch.zeros_like(v) for k, v in self.model.state_dict().items()}  # Initialize global control variate
        self.__dict__.update(kwargs)

    def select_clients(self):
        # Randomly select a subset of clients based on sample_fraction
        num_selected_clients = max(1, int(self.num_clients * self.sample_fraction))
        return random.sample(range(self.num_clients), num_selected_clients)

    def update(self, local_states):
        # Get sampled clients for this round
        sampled_clients = self.select_clients()

        # Extract updates only from sampled clients
        delta_y = OrderedDict()
        delta_c = OrderedDict()
        for client_id in sampled_clients:
            state = local_states[client_id]
            delta_y[client_id] = {k: v.to(self.device) for k, v in state['delta_y'].items()}
            delta_c[client_id] = {k: v.to(self.device) for k, v in state['delta_c'].items()}

        # Aggregate updates from sampled clients
        global_delta_y = {k: torch.zeros_like(v) for k, v in self.model.state_dict().items()}
        global_delta_c = {k: torch.zeros_like(v) for k, v in self.global_control.items()}
        for client_id in delta_y:
            for k in global_delta_y:
                global_delta_y[k] += delta_y[client_id][k] * self.weights[client_id]
                global_delta_c[k] += delta_c[client_id][k] * self.weights[client_id]

        # Normalize by the number of sampled clients
        for k in self.model.state_dict().keys():
            global_delta_y[k] /= len(sampled_clients)
            global_delta_c[k] /= len(sampled_clients)
            self.model.state_dict()[k] += self.server_lr * global_delta_y[k]
            self.global_control[k] += global_delta_c[k]

        return self.model.state_dict(), self.global_control
