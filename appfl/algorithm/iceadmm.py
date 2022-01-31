import logging

log = logging.getLogger(__name__)

from collections import OrderedDict
from .algorithm import BaseServer, BaseClient

import torch
from torch.optim import *
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import copy


class ICEADMMServer(BaseServer):
    def __init__(self, weights, model, num_clients, device, **kwargs):
        super(ICEADMMServer, self).__init__(weights, model, num_clients, device)
        self.__dict__.update(kwargs)        
         
    def update(self, local_states: OrderedDict):
        
        """ Inputs for the global model update"""
        global_state = self.model.state_dict()
        super(ICEADMMServer, self).primal_recover_from_local_states(local_states)
        super(ICEADMMServer, self).dual_recover_from_local_states(local_states)
        super(ICEADMMServer, self).penalty_recover_from_local_states(local_states)

        total_penalty = 0
        for i in range(self.num_clients):
            total_penalty += self.penalty[i]

        """ global_state calculation """
        for name, param in self.model.named_parameters():
            tmp = 0.0
            for i in range(self.num_clients):
                ## change device
                self.primal_states[i][name] = self.primal_states[i][name].to(
                    self.device
                )
                self.dual_states[i][name] = self.dual_states[i][name].to(self.device)
                ## computation
                tmp += (self.penalty[i] / total_penalty) * self.primal_states[i][name] + (
                    1.0 / total_penalty
                ) * self.dual_states[i][name]

            global_state[name] = tmp

        """ model update """                
        self.model.load_state_dict(global_state)



class ICEADMMClient(BaseClient):
    def __init__(self, id, weight, model, dataloader, device, **kwargs):
        super(ICEADMMClient, self).__init__(id, weight, model, dataloader, device)
        self.__dict__.update(kwargs)
        self.loss_fn = CrossEntropyLoss()
        
        """ 
        At initial, (1) primal_state = global_state, (2) dual_state = 0
        """  
        self.model.to(device)   
        for name, param in model.named_parameters():                        
            self.primal_state[name] = param.data
            self.dual_state[name] = torch.zeros_like(param.data)

        self.penalty = kwargs['init_penalty']      
        self.proximity = kwargs['init_proximity']      
        self.residual = OrderedDict()
        self.residual['primal'] = 0
        self.residual['dual'] = 0            

    def update(self):

        self.model.train()
        self.model.to(self.device)

        optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)

        """ Inputs for the local model update """         
        global_state = copy.deepcopy(self.model.state_dict())        

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

                ## STEP: Update primal and dual
                coefficient = 1
                if self.coeff_grad == True:
                    coefficient = self.weight * len(target) / len(self.dataloader.dataset)                
                
                self.iceadmm_step(coefficient, global_state)    
 
        """ Differential Privacy  """
        if self.privacy == True:
            super(ICEADMMClient, self).laplace_mechanism_output_perturb()

        """ Update local_state """
        self.local_state = OrderedDict()
        self.local_state["primal"] = copy.deepcopy(self.primal_state)
        self.local_state["dual"] = copy.deepcopy(self.dual_state)                
        self.local_state["penalty"] = OrderedDict()
        self.local_state["penalty"][self.id] = self.penalty

        return self.local_state

    def iceadmm_step(self,coefficient,global_state):
        for name, param in self.model.named_parameters():

            grad = param.grad * coefficient
            ## Update primal
            self.primal_state[name] = self.primal_state[name] - (
                self.penalty * (self.primal_state[name] - global_state[name])
                + grad
                + self.dual_state[name]
            ) / (self.weight * self.proximity + self.penalty)
            ## Update dual
            self.dual_state[name] = self.dual_state[name] + self.penalty * (
                self.primal_state[name] - global_state[name]
            )

