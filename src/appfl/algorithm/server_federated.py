import logging

from collections import OrderedDict
from .algorithm import BaseServer, BaseClient

import torch
from torch.optim import *
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import copy


class FedServer(BaseServer):
    def __init__(self, weights, model, loss_fn, num_clients, device, **kwargs):
        super(FedServer, self).__init__(weights, model, loss_fn, num_clients, device)
        self.__dict__.update(kwargs)
        self.logger = logging.getLogger(__name__)

        self.step = OrderedDict()
        """ Group 1 """
        self.pseudo_grad = OrderedDict()
        self.m_vector = OrderedDict()
        self.v_vector = OrderedDict()
        for name, _ in self.model.named_parameters():
            self.m_vector[name] = torch.zeros_like(self.model.state_dict()[name])
            self.v_vector[name] = (
                torch.zeros_like(self.model.state_dict()[name])
                + self.server_adapt_param
            )
        """ Group 2 """
        self.pseudo_grad_vec = OrderedDict()
        self.model_size = OrderedDict()
        self.approx_H_matrix = OrderedDict()
        for name, _ in self.model.named_parameters():
            self.model_size[name] = self.model.state_dict()[name].size()
            # print("flat size=", torch.flatten(self.model.state_dict()[name]).size())

            ## TODO: Too LARGE (Reduceing gradient may be needed)
            # self.approx_H_matrix[name] = torch.eye( torch.flatten(self.model.state_dict()[name]).size()[0] )
            # print("H_shape=", self.approx_H_matrix[name].shape)

    def update_m_vector(self):
        for name, _ in self.model.named_parameters():
            self.m_vector[name] = (
                self.server_momentum_param_1 * self.m_vector[name]
                + (1.0 - self.server_momentum_param_1) * self.pseudo_grad[name]
            )

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

    def logging_iteration(self, cfg, logger, t):
        if t == 0:
            title = super(FedServer, self).log_title()
            logger.info(title)

        contents = super(FedServer, self).log_contents(cfg, t)
        logger.info(contents)

    def logging_summary(self, cfg, logger):
        super(FedServer, self).log_summary(cfg, logger)
