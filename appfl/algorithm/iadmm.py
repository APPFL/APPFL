import logging

log = logging.getLogger(__name__)

from collections import OrderedDict
from .algorithm import BaseServer, BaseClient

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import copy

class IADMMServer(BaseServer):
    def __init__(self, model, num_clients, device, dataloader=None, **kwargs):
        super(IADMMServer, self).__init__(model, num_clients, device)
        
        self.__dict__.update(kwargs) 

        self.dataloader = dataloader
        if self.dataloader is not None:
            self.loss_fn = CrossEntropyLoss()
        else:
            self.loss_fn = None

        self.num_clients = num_clients

        self.dual_states = OrderedDict()        
        for i in range(num_clients):
            self.dual_states[i] = OrderedDict()        
            for name, param in model.named_parameters():
                self.dual_states[i][name] = torch.zeros_like(param.data)            

    # update global model
    def update(self, global_state: OrderedDict , local_states: OrderedDict):

        # print("global_state=", global_state["fc2.bias"])
        # for i in range(self.num_clients):
        #     print("client=", i, " local_state=", local_states[i]["fc2.bias"])

        ## Update dual        
        # for i in range(self.num_clients):            
            # print("client=", i, " dual=", self.dual_states[i]["fc2.bias"][0], " global=", global_state["fc2.bias"][0], " local=",local_states[i]["fc2.bias"][0])
        for name, param in self.model.named_parameters():            
            for i in range(self.num_clients):
                self.dual_states[i][name] = self.dual_states[i][name] + self.penalty*(global_state[name] - local_states[i][name])


        # for i in range(self.num_clients):
        #     print("client=", i, " dual_next=", self.dual_states[i]["fc2.bias"])

        ## Update global
        for name, param in self.model.named_parameters():
            tmp=0.0
            for i in range(self.num_clients):
                tmp += local_states[i][name] - (1.0/self.penalty) * self.dual_states[i][name] 
            global_state[name] = tmp / self.num_clients

        # print("updated_global_state=", global_state["fc2.bias"])
        self.model.load_state_dict(global_state)
 
    # NOTE: this is only for testing purpose.
    def validation(self):
        if self.loss_fn is None or self.dataloader is None:
            return 0.0, 0.0

        self.model.to(self.device)
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for img, target in self.dataloader:
                img = img.to(self.device)
                target = target.to(self.device)
                logits = self.model(img)
                test_loss += self.loss_fn(logits, target).item()
                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        # FIXME: do we need to sent the model to cpu again?
        # self.model.to("cpu")
        test_loss = test_loss / len(self.dataloader)
        accuracy = 100.0 * correct / len(self.dataloader.dataset)

        return test_loss, accuracy


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

        # print("Sub: id=", self.id, " global_state=", self.global_state["fc2.bias"]) 
        # print("Sub: id=", self.id, " local_state=", self.local_state["fc2.bias"]) 
        
        ## Gradient of the local point
        self.model.load_state_dict(self.local_state)
        for name, param in self.model.named_parameters():            
            self.local_grad[name] = torch.zeros_like(param.data)                
        tmpcnt=0
        for data, target in self.dataloader:                
            tmpcnt+=1
            data = data.to(self.device)
            target = target.to(self.device)                                    
            output = self.model(data)
            loss = self.loss_fn(output, target)
            loss.backward()            
            
            for name, param in self.model.named_parameters():
                self.local_grad[name] += param.grad
 
        # print("Sub: id=", self.id, " local_grad=", self.local_grad["fc2.bias"])
 
        ## Update local
        for name, param in self.model.named_parameters():
            self.local_state[name] = self.global_state[name] + (1.0/self.penalty) * ( self.dual_state[name] - self.local_grad[name] )
        ## Update dual
        # print("Sub: id=", self.id, " dual=", self.dual_state["fc2.bias"][0], " global=", self.global_state["fc2.bias"][0], " local=",self.local_state["fc2.bias"][0])
        for name, param in self.model.named_parameters():
            self.dual_state[name] = self.dual_state[name] + self.penalty*(self.global_state[name] - self.local_state[name])
        
        # print("Sub: id=", self.id, " dual_next=", self.dual_state["fc2.bias"])

        ## Update model
        self.model.load_state_dict(self.local_state)

        # print("Sub: id=", self.id, " next_local_state=", self.local_state["fc2.bias"])  
        # return self.local_state 
        
