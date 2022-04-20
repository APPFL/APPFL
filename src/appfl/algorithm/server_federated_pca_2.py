import logging

from collections import OrderedDict
from .algorithm import BaseServer, BaseClient

import torch
from torch.optim import *
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import copy


class FedServerPCA2(BaseServer):
    def __init__(self, weights, model, num_clients, device, **kwargs):
        super(FedServerPCA2, self).__init__(weights, model, num_clients, device)
        self.__dict__.update(kwargs)
        self.logger = logging.getLogger(__name__)
        
        
    def update(self, local_states: OrderedDict):

        """Inputs for the global model update"""
        self.global_state = copy.deepcopy(self.model.state_dict())
        super(FedServerPCA2, self).param_vec_recover_from_local_states(local_states)
         
        param_vec_avg = 0
        for id in range(self.num_clients):
            self.param_vec[id] = self.param_vec[id].to(self.device)
            param_vec_avg += self.weights[id] * self.param_vec[id]

        return param_vec_avg
 

 

    def logging_iteration(self, cfg, logger, t):
        if t == 0:
            title = super(FedServerPCA2, self).log_title()
            logger.info(title)

        contents = super(FedServerPCA2, self).log_contents(cfg, t)
        logger.info(contents)

    def logging_summary(self, cfg, logger):
        super(FedServerPCA2, self).log_summary(cfg, logger)
 