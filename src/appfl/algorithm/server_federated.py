import abc
import copy
import torch
import logging
from torch.optim import *
from .fl_base import BaseServer
from collections import OrderedDict

class FedServer(BaseServer):
    """
    FedServer:
        Abstract server class of general FL algorithms that aggregates and updates model parameters.
    Args: 
        weights (Dict): aggregation weight assigned to each client
        model (nn.Module): torch neural network model to train
        loss_fn (nn.Module): loss function
        num_clients (int): the number of clients
        device (str): device for computation
    """
    def __init__(self, weights, model, loss_fn, num_clients, device, **kwargs):
        super(FedServer, self).__init__(weights, model, loss_fn, num_clients, device)
        self.__dict__.update(kwargs)
        self.logger = logging.getLogger(__name__)
        self.step = OrderedDict()
        self.list_named_parameters = []
        self.pseudo_grad = OrderedDict()
        if not self.partial_aggregation:
            for name, _ in self.model.named_parameters():
                self.list_named_parameters.append(name)
            if hasattr(self, "server_momentum_param_1"):
                self.m_vector = OrderedDict()
                for name, _ in self.model.named_parameters():
                    self.list_named_parameters.append(name)
                    self.m_vector[name] = torch.zeros_like(self.model.state_dict()[name], device=device)
            if hasattr(self, "server_adapt_param"):
                self.v_vector = OrderedDict()
                for name, _ in self.model.named_parameters():
                    self.v_vector[name] = torch.zeros_like(self.model.state_dict()[name], device=device) + self.server_adapt_param**2

    def update_m_vector(self):
        """Update the `m_vector` in adaptive federated optimization."""
        for name in self.list_named_parameters:
            if not self.model_state_inited:
                self.m_vector[name] = torch.zeros_like(self.model_state_dict[name], device=self.device)
            self.m_vector[name] = self.server_momentum_param_1 * self.m_vector[name] - (1.0 - self.server_momentum_param_1) * self.pseudo_grad[name]

    def compute_pseudo_gradient(self):
        """Compute the gradient from the client local updates, where gradient is the difference between the old model and the new model."""
        for name in self.list_named_parameters:
            self.pseudo_grad[name] = torch.zeros_like(self.model.state_dict()[name] if not self.partial_aggregation else self.model_state_dict[name])
            for i in range(self.num_clients):
                self.pseudo_grad[name] += self.weights[i] * (self.global_state[name] - self.primal_states[i][name])

    @abc.abstractmethod
    def compute_step(self):
        """Compute the step for global model gradient descend."""
        pass

    def update(self, local_states: list):
        # Init model state dict when server does not have model arch initially
        if self.partial_aggregation and not self.model_state_inited:
            for key, value in local_states[0].items():
                self.model_state_dict[key] = torch.zeros_like(value, device=self.device)
                self.list_named_parameters.append(key)

        self.global_state = copy.deepcopy(self.model.state_dict()) if not self.partial_aggregation else self.model_state_dict

        # Load the client local states to the server device
        for sid, states in enumerate(local_states):
            if states is not None:
                self.primal_states[sid] = states
        for i in range(self.num_clients):
            for name in self.primal_states[i]:
                self.primal_states[i][name] = self.primal_states[i][name].to(self.device)

        ## Update global state
        self.compute_step()

        state_dict_keys = self.model.state_dict().keys() if not self.partial_aggregation else self.model_state_dict.keys()
        for name in state_dict_keys:        
            if name in self.list_named_parameters: 
                self.global_state[name] += self.step[name]            
            else:
                tmpsum = torch.zeros_like(self.global_state[name], device=self.device)                
                for i in range(self.num_clients):
                    tmpsum += self.primal_states[i][name]                
                self.global_state[name] = torch.div(tmpsum, self.num_clients)

        if not self.partial_aggregation:
            self.model.load_state_dict(self.global_state)
        else:
            for name in self.list_named_parameters:
                self.model_state_dict[name] = self.global_state[name]
        self.model_state_inited = True

    def logging_iteration(self, cfg, logger, t):
        if t == 0:
            title = super(FedServer, self).log_title()
            logger.info(title)
        contents = super(FedServer, self).log_contents(cfg, t)
        logger.info(contents)
