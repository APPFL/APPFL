import logging

log = logging.getLogger(__name__)

from collections import OrderedDict
from .algorithm import BaseServer, BaseClient

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader


class FedAvgServer(BaseServer):
    def __init__(self, model, num_clients, device, dataloader=None, **kwargs):
        super(FedAvgServer, self).__init__(model, num_clients, device)
        
        self.__dict__.update(kwargs) 

        self.dataloader = dataloader
        if self.dataloader is not None:
            self.loss_fn = CrossEntropyLoss()
        else:
            self.loss_fn = None

    # update global model
    def update(self, global_state: OrderedDict , local_states: OrderedDict):
        
        for k, state in local_states.items():
            for key in self.model.state_dict().keys():
                if key in global_state.keys():
                    global_state[key] += state[key] / self.num_clients
                else:
                    global_state[key] = state[key] / self.num_clients

        self.model.load_state_dict(global_state)             

class FedAvgClient(BaseClient):
    def __init__(
        self, id, model, optimizer, optimizer_args, dataloader, device, **kwargs
    ):
        super(FedAvgClient, self).__init__(
            id, model, optimizer, optimizer_args, dataloader, device
        )
        self.loss_fn = CrossEntropyLoss()
        self.__dict__.update(kwargs)
        self.id = id

    # update local model
    def update(self):
        self.model.train()
        self.model.to(self.device)
        optimizer = self.optimizer(self.model.parameters(), **self.optimizer_args)
 
        for i in range(self.num_local_epochs):
            # log.info(f"[Client ID: {self.id: 03}, Local epoch: {i+1: 04}]")
            
            for data, target in self.dataloader:                
                data = data.to(self.device)
                target = target.to(self.device)                
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()                
                optimizer.step()
 
        # self.model.to("cpu")
