import logging

from collections import OrderedDict
from .algorithm import BaseServer, BaseClient

import torch
from torch.optim import *
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import copy


class FedServerPCA1(BaseServer):
    def __init__(self, weights, model, num_clients, device, **kwargs):
        super(FedServerPCA1, self).__init__(weights, model, num_clients, device)
        self.__dict__.update(kwargs)
        self.logger = logging.getLogger(__name__)
 
        ## construct
        self.P = OrderedDict()
        self.EVR = OrderedDict()
        for id in range(self.num_clients):
            self.P[id], self.EVR[id] = super(FedServerPCA1, self).construct_projection_matrix(id)
        
        
        
    def update(self, local_states: OrderedDict):

        """Inputs for the global model update"""
        self.global_state = copy.deepcopy(self.model.state_dict())
        super(FedServerPCA1, self).param_vec_recover_from_local_states(local_states)
         
        param_vec_avg = 0
        for id in range(self.num_clients):
            self.param_vec[id] = self.param_vec[id].to(self.device)

            print("server_param_vec_red=", self.param_vec[id].shape)
        
            ## back to original space
            self.param_vec[id] = torch.mm(self.P[id].transpose(0, 1), self.param_vec[id])

            print("server_param_vec=", self.param_vec[id].shape)
            
            param_vec_avg += (1.0/self.num_clients) * self.param_vec[id]


        print("Before=", self.model.state_dict()["linear.bias"])
        print("------")
        print(param_vec_avg)
        super(FedServerPCA1, self).update_param(param_vec_avg)             

        
        print("After=", self.model.state_dict()["linear.bias"])


        idx = 0
        for name, param in self.model.named_parameters():
            arr_shape = param.data.shape
            size = 1
            for i in range(len(list(arr_shape))):
                size *= arr_shape[i]
            self.global_state[name] = param_vec_avg[idx:idx+size].reshape(arr_shape)
            idx += size

        

        """ model update """
        self.model.load_state_dict(self.global_state)

        print("22After=", self.model.state_dict()["linear.bias"])
        
 

    def logging_iteration(self, cfg, logger, t):
        if t == 0:
            title = super(FedServerPCA1, self).log_title()
            logger.info(title)

        contents = super(FedServerPCA1, self).log_contents(cfg, t)
        logger.info(contents)

    def logging_summary(self, cfg, logger):
        super(FedServerPCA1, self).log_summary(cfg, logger)
 