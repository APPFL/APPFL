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


class ICEADMMServer(BaseServer):
    def __init__(self, weights, model, num_clients, device, **kwargs):
        super(ICEADMMServer, self).__init__(weights, model, num_clients, device)

        self.__dict__.update(kwargs)
        self.num_clients = num_clients
        self.weights = weights

        self.primal_states = OrderedDict()
        self.dual_states = OrderedDict()
        for i in range(num_clients):
            self.primal_states[i] = OrderedDict()
            self.dual_states[i] = OrderedDict()

        self.penalty = OrderedDict()

    def update(self, local_states: OrderedDict):
        
        """Inputs"""
        global_state = self.model.state_dict()
        primal_recover_from_local_states(self, local_states)
        dual_recover_from_local_states(self, local_states)
        penalty_recover_from_local_states(self, local_states)        

        total_penalty = 0
        for i in range(self.num_clients):
            total_penalty += self.penalty[i]

        """ Outputs """
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
        self.proximity = kwargs['proximity']      
        self.residual = OrderedDict()
        self.residual['primal'] = 0
        self.residual['dual'] = 0            

    def update(self, cid, model_info):

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

                ## STEP: Update primal and dual
                coefficient = 1
                if self.coeff_grad == True:
                    coefficient = self.weight * len(target) / len(self.dataloader.dataset)

                iceadmm_step(self, coefficient)


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
        self.local_state["dual"] = OrderedDict()
        for name, param in self.model.named_parameters():
            self.local_state["primal"][name] = copy.deepcopy(self.primal_state[name])
            self.local_state["dual"][name] = copy.deepcopy(self.dual_state[name])
        
        self.local_state["penalty"] = OrderedDict()
        self.local_state["penalty"][self.id] = self.penalty

        return self.local_state
