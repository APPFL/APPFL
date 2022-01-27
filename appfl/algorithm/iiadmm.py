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
import math


class IIADMMServer(BaseServer):
    def __init__(self, weights, model, num_clients, device, **kwargs):
        super(IIADMMServer, self).__init__(weights, model, num_clients, device)

        self.__dict__.update(kwargs)
        self.num_clients = num_clients
        self.weights = weights

        self.primal_states = OrderedDict()
        self.dual_states = OrderedDict()
        for i in range(num_clients):
            self.primal_states[i] = OrderedDict()
            self.dual_states[i] = OrderedDict()
            ## dual computation at local
            for name, param in model.named_parameters():
                self.dual_states[i][name] = torch.zeros_like(param.data)
        
        self.penalty = OrderedDict()

    def update(self, local_states: OrderedDict):
    
        """ Inputs """
        global_state = self.model.state_dict()
        primal_recover_from_local_states(self, local_states)        
        penalty_recover_from_local_states(self, local_states)        

        # for i in range(self.num_clients):
        #     print("self.penalty[",i,"]=", self.penalty[i])

        """ Outputs """
        for name, param in self.model.named_parameters():
            tmp = 0.0
            for i in range(self.num_clients):

                ## change device
                self.primal_states[i][name] = self.primal_states[i][name].to(
                    self.device
                )
                ## dual
                self.dual_states[i][name] = self.dual_states[i][name] + self.penalty[i] * (
                    global_state[name] - self.primal_states[i][name]
                )
                ## computation
                tmp += (
                    self.primal_states[i][name]
                    - (1.0 / self.penalty[i]) * self.dual_states[i][name]
                )

            global_state[name] = tmp / self.num_clients

        """ model update """        
        self.model.load_state_dict(global_state)
 


class IIADMMClient(BaseClient):
    def __init__(self, id, weight, model, dataloader, device, **kwargs):
        super(IIADMMClient, self).__init__(id, weight, model, dataloader, device)

        self.loss_fn = CrossEntropyLoss()
        self.__dict__.update(kwargs)
        self.id = id

        self.model.to(device)                
        self.global_state = OrderedDict()
        self.primal_state = OrderedDict()
        self.dual_state = OrderedDict()
        for name, param in model.named_parameters():            
            self.primal_state[name] = param.data
            self.dual_state[name] = torch.zeros_like(param.data)
        
        self.penalty = kwargs['penalty']        
        self.residual = OrderedDict()
        self.residual['primal'] = 0
        self.residual['dual'] = 0

    def update(self):

        self.model.train()
        self.model.to(self.device)

        optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)

        """ Inputs """        
        for name, param in self.model.named_parameters():
            self.global_state[name] = copy.deepcopy(param.data)
        
        
        ## TODO: residual_calculation + adaptive_penalty
        ## Option 1: change penalty for every comm. round
        if self.residual['primal'] == 0 and self.residual['dual'] == 0:
            self.penalty = self.penalty
        # else:


        """ Multiple local update """
        for i in range(self.num_local_epochs):
            for data, target in self.dataloader:           

                self.model.load_state_dict(self.primal_state)

                ## TODO: residual_calculation + adaptive_penalty 
                ## Option 2: change penalty for every local iteration                

                data = data.to(self.device)
                target = target.to(self.device)

                if self.accum_grad == False:
                    optimizer.zero_grad()

                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()

                ## STEP: Update primal
                coefficient = 1
                if self.coeff_grad == True:
                    
                    # coefficient = (
                    #     self.weight * len(target) / len(self.dataloader.dataset)
                    # )

                    coefficient = self.weight  ## NOTE: BATCH + FEMNIST, rho=0.07

                iiadmm_step(self, coefficient, optimizer)
 

        ## Update dual
        for name, param in self.model.named_parameters():
            self.dual_state[name] = self.dual_state[name] + self.penalty * (
                self.global_state[name] - self.primal_state[name]
            )

        """ Differential Privacy """
        if self.privacy == True:
            # Note: Scale_value = Sensitivity_value / self.epsilon

            Scale_value = self.scale_value

            for name, param in self.model.named_parameters():
                mean = torch.zeros_like(param.data)
                scale = torch.zeros_like(param.data) + Scale_value
                m = torch.distributions.laplace.Laplace(mean, scale)
                self.primal_state[name] += m.sample()

        """ Update local_state """
        self.local_state = OrderedDict()
        
        self.local_state["primal"] = OrderedDict()        
        for name, param in self.model.named_parameters():
            self.local_state["primal"][name] = copy.deepcopy(self.primal_state[name])
        
        self.local_state["penalty"] = OrderedDict()
        self.local_state["penalty"][self.id] = self.penalty

        return self.local_state
