import logging

log = logging.getLogger(__name__)

from collections import OrderedDict
from .algorithm import BaseServer, BaseClient

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import copy

class IADMMServer(BaseServer):
    def __init__(self, model, num_clients, device, **kwargs):
        super(IADMMServer, self).__init__(model, num_clients, device)
        
        self.__dict__.update(kwargs) 

        self.num_clients = num_clients        

        self.dual_states = OrderedDict()        
        for i in range(num_clients):
            self.dual_states[i] = OrderedDict()        
            for name, param in model.named_parameters():
                self.dual_states[i][name] = torch.zeros_like(param.data)            

    # update global model
    def update(self, global_state: OrderedDict , local_states: OrderedDict):

        ## Update dual                
        for name, param in self.model.named_parameters():            
            for i in range(self.num_clients):
                self.dual_states[i][name] = self.dual_states[i][name] + self.penalty*(global_state[name] - local_states[i][name])

        ## Update global
        for name, param in self.model.named_parameters():
            tmp=0.0
            for i in range(self.num_clients):
                tmp += local_states[i][name] - (1.0/self.penalty) * self.dual_states[i][name] 
            global_state[name] = tmp / self.num_clients

        self.model.load_state_dict(global_state)
 

class IADMMClient(BaseClient):
    def __init__(
        self, id, model, optimizer, optimizer_args, dataloader, device, **kwargs
    ):
        super(IADMMClient, self).__init__(
            id, model, optimizer, optimizer_args, dataloader, device
        )
        
        self.loss_fn = CrossEntropyLoss()
        self.__dict__.update(kwargs)        

        self.id = id
        
        self.model.to(device)
        self.global_state = OrderedDict()        
        self.local_state = OrderedDict()
        self.dual_state = OrderedDict()        
        self.local_grad = OrderedDict()
        for name, param in model.named_parameters():
            self.global_state[name] = param.data
            self.local_state[name] = param.data
            self.dual_state[name] = torch.zeros_like(param.data)            
            self.local_grad[name] = torch.zeros_like(param.data)
            
 
    # update local model
    def update(self):
        self.model.train()
        self.model.to(self.device)

        ## Global state
        for name, param in self.model.named_parameters():
            self.global_state[name] = copy.deepcopy(param.data)        
        
        for i in range(self.num_local_epochs):
            # log.info(f"[Client ID: {self.id: 03}, Local epoch: {i+1: 04}]")
            ## Gradient of the local point
            self.model.load_state_dict(self.local_state)
            for name, param in self.model.named_parameters():            
                self.local_grad[name] = torch.zeros_like(param.data)                
             
            for data, target in self.dataloader:                                
                data = data.to(self.device)
                target = target.to(self.device)                                    
                output = self.model(data)
                loss = self.loss_fn(output, target)                
                loss.backward()            
             
                for name, param in self.model.named_parameters():
                    self.local_grad[name] = param.grad
            
            ## Update local            
            for name, param in self.model.named_parameters():                
                self.local_state[name] = self.global_state[name] + (1.0/self.penalty) * ( self.dual_state[name] - self.local_grad[name] )
        
        ## Update dual        
        for name, param in self.model.named_parameters():
            self.dual_state[name] = self.dual_state[name] + self.penalty*(self.global_state[name] - self.local_state[name])
        
        ## Update model
        self.model.load_state_dict(self.local_state)

        
