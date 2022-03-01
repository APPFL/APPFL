import logging

log = logging.getLogger(__name__)

from collections import OrderedDict
from .algorithm import BaseServer, BaseClient

import torch
from torch.optim import *
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import copy


class FedServer(BaseServer):
    def __init__(self, weights, model, num_clients, device, **kwargs):
        super(FedServer, self).__init__(weights, model, num_clients, device)
        self.__dict__.update(kwargs)
        self.logger = logging.getLogger(__name__)

        self.pseudo_grad = OrderedDict()        
        self.step = OrderedDict()
        
        self.m_vector = OrderedDict()
        self.v_vector = OrderedDict()
        for name, _ in self.model.named_parameters():
            self.m_vector[name] = torch.zeros_like(self.model.state_dict()[name])
            self.v_vector[name] = torch.zeros_like(self.model.state_dict()[name]) + self.server_adapt_param

    def update_m_vector(self):
        for name, _ in self.model.named_parameters():                        
            self.m_vector[name] = self.server_momentum_param_1 * self.m_vector[name] + (1.0 - self.server_momentum_param_1) * self.pseudo_grad[name]   

    # def compute_pseudo_gradient(self):        
    #     for name, _ in self.model.named_parameters():
    #         self.pseudo_grad[name] = torch.zeros_like(self.model.state_dict()[name])
    #         for i in range(self.num_clients):        
    #             self.pseudo_grad[name] += self.primal_states[i][name] - global_state[name]
    #         self.pseudo_grad[name] /= self.num_clients

            

    def update(self, local_states: OrderedDict):

        """ Inputs for the global model update """
        self.global_state = copy.deepcopy(self.model.state_dict())
        super(FedServer, self).primal_recover_from_local_states(local_states)

        """ residual calculation """
        prim_res = super(FedServer, self).primal_residual_at_server()        

        """ change device """
        for i in range(self.num_clients):
            for name, _ in self.model.named_parameters():
                self.primal_states[i][name] = self.primal_states[i][name].to(self.device)

        """ global_state calculation """                
        self.compute_step()
        for name, _ in self.model.named_parameters():                        
            global_state[name] += self.step[name]
 
        """ model update """
        self.model.load_state_dict(global_state)

        return prim_res, 0, 0, 0
 
        

