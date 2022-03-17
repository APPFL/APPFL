import logging

log = logging.getLogger(__name__)

from collections import OrderedDict
from .algorithm import BaseServer, BaseClient

import torch
from torch.optim import *
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import copy


class FedAvgServer(BaseServer):
    def __init__(self, weights, model, num_clients, device, **kwargs):
        super(FedAvgServer, self).__init__(weights, model, num_clients, device)
        self.__dict__.update(kwargs)
        self.logger = logging.getLogger(__name__)

    def update(self, local_states: OrderedDict):

        """ Inputs for the global model update """
        global_state = copy.deepcopy(self.model.state_dict())
        super(FedAvgServer, self).primal_recover_from_local_states(local_states)

        """ residual calculation """
        prim_res = super(FedAvgServer, self).primal_residual_at_server(global_state)        

        """ global_state calculation """
        self.logger.debug(f"self.weights: {self.weights}")
        for name, _ in self.model.named_parameters():
            tmp = 0.0
            for i in range(self.num_clients):
                ## change device
                self.primal_states[i][name] = self.primal_states[i][name].to(self.device)
                ## computation
                tmp += self.weights[i] * self.primal_states[i][name]

            global_state[name] = tmp


        """ model update """
        self.model.load_state_dict(global_state)

        return prim_res, 0, 0, 0

class FedAvgClient(BaseClient):
    def __init__(self, id, weight, model, dataloader, device, **kwargs):
        super(FedAvgClient, self).__init__(id, weight, model, dataloader, device)
        self.__dict__.update(kwargs)
        self.loss_fn = eval(self.loss_type)

    def update(self):

        self.model.train()
        self.model.to(self.device)

        optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)

        """ Inputs for the local model update """
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

                if self.clip_value != False:                                              
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value, norm_type=self.clip_norm)                

                optimizer.step()

        self.primal_state = copy.deepcopy(self.model.state_dict()) 

        """ Differential Privacy  """
        if self.epsilon != False:
            sensitivity = 0
            if self.clip_value != False:                           
                sensitivity = 2.0 * self.clip_value * self.optim_args.lr 
            scale_value = sensitivity / self.epsilon            
            super(FedAvgClient, self).laplace_mechanism_output_perturb(scale_value)

        """ Update local_state """
        self.local_state = OrderedDict()
        self.local_state["primal"] = copy.deepcopy(self.primal_state)
        self.local_state["dual"] = OrderedDict()
        self.local_state["penalty"] = OrderedDict()
        self.local_state["penalty"][self.id] = 0.0

        return self.local_state
