import logging

log = logging.getLogger(__name__)

from collections import OrderedDict
from .algorithm import BaseServer, BaseClient

import torch
from torch.optim import *
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import copy


class FedAdaptServer(BaseServer):
    def __init__(self, weights, model, num_clients, device, **kwargs):
        super(FedAdaptServer, self).__init__(weights, model, num_clients, device)
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
            

    def update(self, local_states: OrderedDict):

        """ Inputs for the global model update """
        global_state = copy.deepcopy(self.model.state_dict())
        super(FedAdaptServer, self).primal_recover_from_local_states(local_states)

        """ residual calculation """
        prim_res = super(FedAdaptServer, self).primal_residual_at_server(global_state)        

        """ change device """
        for i in range(self.num_clients):
            for name, _ in self.model.named_parameters():
                self.primal_states[i][name] = self.primal_states[i][name].to(self.device)

        """ pseudo gradient """
        for name, _ in self.model.named_parameters():
            self.pseudo_grad[name] = torch.zeros_like(global_state[name])
            for i in range(self.num_clients):        
                self.pseudo_grad[name] += self.primal_states[i][name] - global_state[name]
            self.pseudo_grad[name] /= self.num_clients

        """ global_state calculation """                
        self.compute_step()
        for name, _ in self.model.named_parameters():                        
            global_state[name] += self.step[name]
 
        """ model update """
        self.model.load_state_dict(global_state)

        return prim_res, 0, 0, 0


class FedAvgServer(FedAdaptServer):
    def compute_step(self):        
        for name, _ in self.model.named_parameters():                        
            self.step[name] = self.pseudo_grad[name]        
        
class FedAvgMServer(FedAdaptServer):
    def compute_step(self):        
        super(FedAvgMServer, self).update_m_vector()
        for name, _ in self.model.named_parameters():                        
            self.step[name] = self.m_vector[name]

class FedAdagradServer(FedAdaptServer):
    def compute_step(self):        
        super(FedAdagradServer, self).update_m_vector()
        for name, _ in self.model.named_parameters():    
            self.v_vector[name] = self.v_vector[name] + torch.square(self.pseudo_grad[name])
            self.step[name] = torch.div( self.server_learning_rate * self.m_vector[name], torch.sqrt(self.v_vector[name]) + self.server_adapt_param )

class FedAdamServer(FedAdaptServer):
    def compute_step(self):        
        super(FedAdamServer, self).update_m_vector()
        for name, _ in self.model.named_parameters():    
            self.v_vector[name] = self.server_momentum_param_2 * self.v_vector[name] + (1.0-self.server_momentum_param_2) * torch.square(self.pseudo_grad[name])
            self.step[name] = torch.div( self.server_learning_rate * self.m_vector[name], torch.sqrt(self.v_vector[name]) + self.server_adapt_param )

class FedYogiServer(FedAdaptServer):
    def compute_step(self):        
        super(FedYogiServer, self).update_m_vector()
        for name, _ in self.model.named_parameters():    
            self.v_vector[name] = self.v_vector[name] - (1.0-self.server_momentum_param_2) * torch.mul( torch.square(self.pseudo_grad[name]), torch.sign(self.v_vector[name] - torch.square(self.pseudo_grad[name]) ) )            

            self.step[name] = torch.div( self.server_learning_rate * self.m_vector[name], torch.sqrt(self.v_vector[name]) + self.server_adapt_param )

class FedAdaptClient(BaseClient):
    def __init__(self, id, weight, model, dataloader, device, **kwargs):
        super(FedAdaptClient, self).__init__(id, weight, model, dataloader, device)
        self.__dict__.update(kwargs)
        self.loss_fn = CrossEntropyLoss()

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
            super(FedAdaptClient, self).laplace_mechanism_output_perturb(scale_value)

        """ Update local_state """
        self.local_state = OrderedDict()
        self.local_state["primal"] = copy.deepcopy(self.primal_state)
        self.local_state["dual"] = OrderedDict()
        self.local_state["penalty"] = OrderedDict()
        self.local_state["penalty"][self.id] = 0.0

        return self.local_state


