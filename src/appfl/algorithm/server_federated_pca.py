import logging

from collections import OrderedDict
from .algorithm import BaseServer, BaseClient

import torch
from torch.optim import *
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import copy
import numpy as np

class FedServerPCA(BaseServer):
    def __init__(self, weights, model, num_clients, device, **kwargs):
        super(FedServerPCA, self).__init__(weights, model, num_clients, device)
        self.__dict__.update(kwargs)
        self.logger = logging.getLogger(__name__)

        self.step = None 
        self.pseudo_grad = None
        self.m_vector = None         
        self.v_vector = None


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

        ## construct projection        
        self.P, self.EVR = super(FedServerPCA, self).construct_projection_matrix()            

    def update_m_vector(self):

        self.m_vector = self.server_momentum_param_1 * self.m_vector + (1.0 - self.server_momentum_param_1) * self.pseudo_grad
 
              
    def compute_pseudo_gradient(self):

        self.pseudo_grad = 0
               
        for id in range(self.num_clients):             
            self.pseudo_grad += self.weights[id] * self.reduced_grad_vec[id]
 

    def update(self, local_states: OrderedDict):

        """Inputs for the global model update""" 

        global_state_vec = super(FedServerPCA, self).get_model_param_vec()        
        global_state_vec = torch.tensor(global_state_vec, device = self.device)
        global_state_vec = global_state_vec.reshape(-1,1)
 
        super(FedServerPCA, self).grad_vec_recover_from_local_states(local_states)
        
 
        self.compute_step()
        global_state_vec += self.step
 
        super(FedServerPCA, self).update_param(global_state_vec)
 

    def logging_iteration(self, cfg, logger, t):
        if t == 0:
            title = super(FedServerPCA, self).log_title()
            logger.info(title)

        contents = super(FedServerPCA, self).log_contents(cfg, t)
        logger.info(contents)

    def logging_summary(self, cfg, logger):
        super(FedServerPCA, self).log_summary(cfg, logger)
