import logging

log = logging.getLogger(__name__)

from collections import OrderedDict
from .algorithm import BaseServer, BaseClient
from .misc import *

import torch
from torch.optim import *
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import copy


class FedAvgServer(BaseServer):
    def __init__(self, weights, model, num_clients, device, **kwargs):
        super(FedAvgServer, self).__init__(weights, model, num_clients, device)

        self.__dict__.update(kwargs)
        self.num_clients = num_clients
        self.weights = weights

        self.primal_states = OrderedDict()
        for i in range(num_clients):
            self.primal_states[i] = OrderedDict()

    def update(self, local_states: OrderedDict):

        """ Inputs """
        global_state = OrderedDict()
        primal_recover_from_local_states(self, local_states)
        

        """ Outputs """
        for name, param in self.model.named_parameters():
            tmp = 0.0
            for i in range(self.num_clients):
                ## change device
                self.primal_states[i][name] = self.primal_states[i][name].to(self.device)
                ## computation
                tmp += self.weights[i] * self.primal_states[i][name]

            global_state[name] = tmp

        """ model update """
        self.model.load_state_dict(global_state)



class FedAvgClient(BaseClient):
    def __init__(self, id, weight, model, dataloader, device, **kwargs):
        super(FedAvgClient, self).__init__(id, weight, model, dataloader, device)
        self.loss_fn = CrossEntropyLoss()
        self.__dict__.update(kwargs)
        self.id = id
 
    def update(self):

        self.model.train()
        self.model.to(self.device)

        optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)

        """ Inputs """                
        ## "global_state" from a server is already stored in 'self.model'
        
        """ Multiple local update """
        for i in range(self.num_local_epochs):
            for data, target in self.dataloader:

                data = data.to(self.device)
                target = target.to(self.device)

                optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()

                optimizer.step()

        """ Differential Privacy """
        if self.privacy == True:
            # Note: Scale_value = Sensitivity_value / self.epsilon

            Scale_value = self.scale_value

            for name, param in self.model.named_parameters():
                mean = torch.zeros_like(param.data)
                scale = torch.zeros_like(param.data) + Scale_value
                m = torch.distributions.laplace.Laplace(mean, scale)
                param.data += m.sample()

        """ Update local_state """
        self.local_state = OrderedDict()
        self.local_state["primal"] = OrderedDict()
        for name, param in self.model.named_parameters():
            self.local_state["primal"][name] = copy.deepcopy(param.data)

        return self.local_state
