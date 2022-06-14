import logging

from collections import OrderedDict
from .algorithm import BaseServer, BaseClient

import torch
from torch.optim import *
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import copy

class AsyncFedServer(FedServer):
    def __init__(self, weights, model, loss_fn, num_clients, device, **kwargs):
        super(FedServer, self).__init__(weights, model, loss_fn, num_clients, device)
    
    def compute_pseudo_gradient(self):
        for name, _ in self.model.named_parameters():
            self.pseudo_grad[name] = torch.zeros_like(self.model.state_dict()[name])
            for i in range(self.num_clients):
                self.pseudo_grad[name] += self.weights[i] * (
                    self.global_state[name] - self.primal_states[i][name]
                )

    def update(self, local_states: OrderedDict):

        """Inputs for the global model update"""
        self.global_state = copy.deepcopy(self.model.state_dict())
        super(FedServer, self).primal_recover_from_local_states(local_states)

        """ residual calculation """
        super(FedServer, self).primal_residual_at_server()

        """ change device """
        for i in range(self.num_clients):
            for name, _ in self.model.named_parameters():
                self.primal_states[i][name] = self.primal_states[i][name].to(
                    self.device
                )

        """ global_state calculation """
        self.compute_step()
        for name, _ in self.model.named_parameters():
            self.global_state[name] += self.step[name]

        """ model update """
        self.model.load_state_dict(self.global_state)
