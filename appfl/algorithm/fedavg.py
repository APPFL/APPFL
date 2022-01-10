import logging

log = logging.getLogger(__name__)

from collections import OrderedDict
from .algorithm import BaseServer, BaseClient

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

class FedAvgServer(BaseServer):
    def __init__(self, model, num_clients, device, **kwargs):
        super(FedAvgServer, self).__init__(model, num_clients, device)

        self.__dict__.update(kwargs)

    # update global model
    def update(self, global_state: OrderedDict, local_states: OrderedDict):
        update_state = OrderedDict()
        for k, state in local_states.items():
            for key in self.model.state_dict().keys():
                if key in update_state.keys():
                    update_state[key] += state[key] / self.num_clients
                else:
                    update_state[key] = state[key] / self.num_clients

        self.model.load_state_dict(update_state)


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
        
        ## Differential Privacy
        if self.privacy == True:            
            # Note: Scale_value = Sensitivity_value / self.epsilon            
            
            Scale_value = self.scale_value             

            for name, param in self.model.named_parameters():                       
                mean  = torch.zeros_like(param.data)
                scale = torch.zeros_like(param.data) + Scale_value
                m = torch.distributions.laplace.Laplace( mean, scale )                    
                param.data += m.sample()         

        # self.model.to("cpu")


