import logging

log = logging.getLogger(__name__)

from collections import OrderedDict
from .algorithm import BaseServer, BaseClient

import torch
from torch.optim import *
# from torch.nn import CrossEntropyLoss, BCELoss
from torch.utils.data import DataLoader
import copy

 
class ClientOptim(BaseClient):
    def __init__(self, id, weight, model, dataloader, device, **kwargs):
        super(ClientOptim, self).__init__(id, weight, model, dataloader, device)
        self.__dict__.update(kwargs)

        self.loss_fn = eval(self.loss_type)
         
        self.train_loss = []

    def update(self):

        self.model.train()
        self.model.to(self.device)

        optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)

        """ Inputs for the local model update 
            "global_state" from a server is already stored in 'self.model'
        """        

        """ Multiple local update """
        for i in range(self.num_local_epochs):
            train_loss = 0; tmpcnt= 0
            for data, target in self.dataloader:

                data = data.to(self.device)
                target = target.to(self.device)
                
                output = self.model(data)
                
                ## TODO: different functions take different formats of arguments
                if self.loss_type == "torch.nn.BCELoss()":
                    target = target.to(torch.float32)                                
                    loss = self.loss_fn(output, target.reshape(-1,1))
                else:
                    loss = self.loss_fn(output, target)            

                optimizer.zero_grad()
                loss.backward()

                train_loss += loss.item()  
                tmpcnt += 1

                if self.clip_value != False:                                              
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value, norm_type=self.clip_norm)                

                optimizer.step()
            train_loss = train_loss/tmpcnt        

        self.train_loss.append(train_loss) 
        
        self.primal_state = copy.deepcopy(self.model.state_dict()) 

        """ Differential Privacy  """
        if self.epsilon != False:
            sensitivity = 0
            if self.clip_value != False:                           
                sensitivity = 2.0 * self.clip_value * self.optim_args.lr 
            scale_value = sensitivity / self.epsilon            
            super(ClientOptim, self).laplace_mechanism_output_perturb(scale_value)

        """ Update local_state """
        self.local_state = OrderedDict()
        self.local_state["primal"] = copy.deepcopy(self.primal_state)
        self.local_state["dual"] = OrderedDict()
        self.local_state["penalty"] = OrderedDict()
        self.local_state["penalty"][self.id] = 0.0        

        return self.local_state


