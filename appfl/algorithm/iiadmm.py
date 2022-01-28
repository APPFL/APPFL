import logging

log = logging.getLogger(__name__)

from collections import OrderedDict
from .algorithm import BaseServer, BaseClient

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
        
        """ 
        At initial, dual_state = 0
        """  
        for i in range(num_clients):                        
            for name, param in model.named_parameters():
                self.dual_states[i][name] = torch.zeros_like(param.data)               

    def update(self, local_states: OrderedDict):
    
        """ Inputs for the global model update"""
        global_state = self.model.state_dict()
        super(IIADMMServer, self).primal_recover_from_local_states(local_states)
        super(IIADMMServer, self).penalty_recover_from_local_states(local_states)        


        """ global_state calculation """
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

                ## STEP: Update primal
                coefficient = 1
                if self.coeff_grad == True:
                    
                    # coefficient = (
                    #     self.weight * len(target) / len(self.dataloader.dataset)
                    # )

                    coefficient = self.weight  ## NOTE: BATCH + FEMNIST, rho=0.07

                iiadmm_step(self, coefficient, global_state, optimizer)
 

        ## Update dual
        for name, param in self.model.named_parameters():
            self.dual_state[name] = self.dual_state[name] + self.penalty * (
                global_state[name] - self.primal_state[name]
            )

        """ Differential Privacy  """
        if self.privacy == True:
            super(IIADMMClient, self).laplace_mechanism_output_perturb()

        """ Update local_state """
        self.local_state = OrderedDict()                
        self.local_state["primal"] = copy.deepcopy(self.primal_state)                        
        self.local_state["penalty"] = OrderedDict()
        self.local_state["penalty"][self.id] = self.penalty

        return self.local_state


def optimizer_setting(self):
    momentum = 0
    if "momentum" in self.optim_args.keys():
        momentum = self.optim_args.momentum
    weight_decay = 0
    if "weight_decay" in self.optim_args.keys():
        weight_decay = self.optim_args.weight_decay
    dampening = 0
    if "dampening" in self.optim_args.keys():
        dampening = self.optim_args.dampening
    nesterov = False

    return momentum, weight_decay, dampening, nesterov


def iiadmm_step(self, coefficient, global_state, optimizer):

    momentum, weight_decay, dampening, nesterov = optimizer_setting(self)

    for name, param in self.model.named_parameters():

        grad = copy.deepcopy(param.grad * coefficient)

        if weight_decay != 0:
            grad.add_(weight_decay, self.primal_state[name])
        if momentum != 0:
            param_state = optimizer.state[param]
            if "momentum_buffer" not in param_state:
                buf = param_state["momentum_buffer"] = grad.clone()
            else:
                buf = param_state["momentum_buffer"]
                buf.mul_(momentum).add_(1 - dampening, grad)
            if nesterov:
                grad = self.grad[name].add(momentum, buf)
            else:
                grad = buf

        ## Update primal
        self.primal_state[name] = global_state[name] + (1.0 / self.penalty) * (
            self.dual_state[name] - grad
        )


