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
            



    def initial_model_info(self, comm_size, num_client_groups):

        model_info = OrderedDict()
        for rank in range(1,comm_size):
            model_info[rank] = OrderedDict()
            model_info[rank]['global_state'] = copy.deepcopy(self.model.state_dict())
            model_info[rank]['penalty'] = OrderedDict()
            for _, cid in enumerate(num_client_groups[rank-1]):
                model_info[rank]['penalty'][cid]= self.penalty 

        return model_info
    
    def update(self, t, comm_size, num_client_groups, model_info: OrderedDict, local_states: OrderedDict):

        primal_recover_from_local_states(self, local_states)
        dual_recover_from_local_states(self, local_states)
        global_state = self.model.state_dict()
  
        total_penalty = 0; penalty = {}
        for rank in range(1,comm_size):            
            for _, cid in enumerate(num_client_groups[rank-1]):
                penalty[cid] = model_info[rank]['penalty'][cid] 
                total_penalty += model_info[rank]['penalty'][cid]                           
         
        for name, param in self.model.named_parameters():
            tmp = 0.0
            for i in range(self.num_clients):    
                self.primal_states[i][name] = self.primal_states[i][name].to(self.device)
                self.dual_states[i][name] = self.dual_states[i][name].to(self.device)                

                tmp += (penalty[i]/total_penalty) * self.primal_states[i][name] + (1.0/total_penalty) * self.dual_states[i][name]                
 
            global_state[name] = tmp
        
        ## model correction
        self.model.load_state_dict(global_state)

        for rank in range(1,comm_size):            
            model_info[rank]['global_state'] = copy.deepcopy(global_state)      
        
        return model_info

class ICEADMMClient(BaseClient):
    def __init__(
        self, id, weight, model, dataloader, device, **kwargs
    ):
        super(ICEADMMClient, self).__init__(
            id, weight, model, dataloader, device
        )

        self.loss_fn = CrossEntropyLoss()
        self.__dict__.update(kwargs)
        self.id = id

        self.model.to(device)
        self.global_state = OrderedDict()        
        self.primal_state = OrderedDict()
        self.dual_state = OrderedDict()        
        for name, param in model.named_parameters():
            self.global_state[name] = param.data            
            self.primal_state[name] = param.data            
            self.dual_state[name] = torch.zeros_like(param.data)            
    
    def update(self, cid, model_info):

        self.model.train()
        self.model.to(self.device)

        optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)
        
        penalty = model_info['penalty'][cid]
        r = self.proximity
        
        ## Fix global state            
        for name, param in self.model.named_parameters():
            self.global_state[name] = copy.deepcopy(model_info['global_state'][name].to(self.device))

        ## Multiple local update
        for i in range(self.num_local_epochs):            
            for data, target in self.dataloader:
                
                self.model.load_state_dict(self.primal_state)            
                
                data = data.to(self.device)
                target = target.to(self.device)
                
                if self.accum_grad == False:
                    optimizer.zero_grad()         

                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()
                
                ## STEP: Update primal and dual
                coefficient = self.weight * len(target) / len(self.dataloader.dataset)
                
                for name, param in self.model.named_parameters():
                    
                    grad = param.grad * coefficient

                    self.primal_state[name] = self.primal_state[name] - ( penalty*(self.primal_state[name]-self.global_state[name]) + grad + self.dual_state[name]) /  (self.weight * r + penalty)

                    self.dual_state[name] = self.dual_state[name] + penalty*(self.primal_state[name]-self.global_state[name])
                
                    

                    
        ## Differential Privacy
        if self.privacy == True:            
            # Note: Scale_value = Sensitivity_value / self.epsilon            
            
            Scale_value = self.scale_value             

            for name, param in self.model.named_parameters():       
                mean  = torch.zeros_like(param.data)
                scale = torch.zeros_like(param.data) + Scale_value
                m = torch.distributions.laplace.Laplace( mean, scale )   
                self.primal_state[name] += m.sample()                            
     
        ## Update local_state
        self.local_state = OrderedDict()
        self.local_state['primal'] = OrderedDict()
        self.local_state['dual'] = OrderedDict()        
        for name, param in self.model.named_parameters():
            self.local_state['primal'][name] = copy.deepcopy(self.primal_state[name])
            self.local_state['dual'][name] = copy.deepcopy(self.dual_state[name])         

        return self.local_state
 