import logging
import os
from collections import OrderedDict
from .algorithm import BaseClient

import torch
from torch.optim import *

from torch.utils.data import DataLoader
import copy

import numpy as np
import time


class ClientSdlbfgs(BaseClient):
    def __init__(
        self, id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader, **kwargs
    ):
        super(ClientSdlbfgs, self).__init__(
            id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader
        )
        self.__dict__.update(kwargs)

        self.round = 0
        
        self.prev_images = torch.tensor([])
        self.prev_labels = torch.tensor([]) 
        for i, (images, labels) in enumerate(dataloader):
            if i == 0:
                images = images.to(self.cfg.device)
                labels = labels.to(self.cfg.device)
                self.prev_images = copy.deepcopy(images)
                self.prev_labels = copy.deepcopy(labels)
            else:
                break
        
        
        self.prev_grad = OrderedDict()
        self.prev_data = OrderedDict()
        self.s_vec =OrderedDict()
        self.y_vec =OrderedDict()
        self.grad =OrderedDict()

        super(ClientSdlbfgs, self).client_log_title()

    def update(self):
        
        """Inputs for the local model update"""

        self.model.to(self.cfg.device)
        

        optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)
 
        """ initial point  """
        optimizer.zero_grad()
        outputs = self.model(self.prev_images)    
        loss = self.loss_fn(outputs, self.prev_labels) 
        loss.backward() 
 
        for name, p in self.model.named_parameters():
            self.prev_data[name]=copy.deepcopy(p.data.reshape(-1))
            self.prev_grad[name]=copy.deepcopy(p.grad.reshape(-1))


        """ Multiple local update """
        start_time=time.time()
        ## initial evaluation
        if self.cfg.validation == True and self.test_dataloader != None:
            test_loss, test_accuracy = super(ClientSdlbfgs, self).client_validation(
                self.test_dataloader
            )
            per_iter_time = time.time() - start_time
            super(ClientSdlbfgs, self).client_log_content(
                0, per_iter_time, 0, 0, test_loss, test_accuracy
            )
            ## return to train mode
            self.model.train()        

        ## local training 
        for t in range(self.num_local_epochs):
            start_time=time.time()
            train_loss = 0
            train_correct = 0            
            tmptotal = 0
            for data, target in self.dataloader:                
                tmptotal += len(target)
                data = data.to(self.cfg.device)
                target = target.to(self.cfg.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                if output.shape[1] == 1:
                    pred = torch.round(output)
                else:
                    pred = output.argmax(dim=1, keepdim=True)
                train_correct += pred.eq(target.view_as(pred)).sum().item()

                if self.clip_value != False:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.clip_value,
                        norm_type=self.clip_norm,
                    )
            ## Validation
            train_loss = train_loss / len(self.dataloader)
            train_accuracy = 100.0 * train_correct / tmptotal
            if self.cfg.validation == True and self.test_dataloader != None:
                test_loss, test_accuracy = super(ClientSdlbfgs, self).client_validation(
                    self.test_dataloader
                )
                per_iter_time = time.time() - start_time
                super(ClientSdlbfgs, self).client_log_content(
                    t+1, per_iter_time, train_loss, train_accuracy, test_loss, test_accuracy
                )
                ## return to train mode
                self.model.train()

            ## save model.state_dict()
            if self.cfg.save_model_state_dict == True:
                path = self.cfg.output_dirname + "/client_%s" % (self.id)
                if not os.path.exists(path):
                    os.makedirs(path)
                torch.save(
                    self.model.state_dict(),
                    os.path.join(path, "%s_%s.pt" % (self.round, t)),
                )
 
 
        self.round += 1

        self.primal_state = copy.deepcopy(self.model.state_dict())

        ## store s_vec and grad
        for name, p in self.model.named_parameters():
            self.s_vec[name] = p.data.reshape(-1) - self.prev_data[name]        
            self.grad[name]  = p.grad.reshape(-1)
 
 
            
        ## gradient of the loss function built upon previous images        
        optimizer.zero_grad()
        outputs = self.model(self.prev_images)
        loss = self.loss_fn(outputs, self.prev_labels) 
        loss.backward() 
        ## store y_vec
        for name, p in self.model.named_parameters():
            self.y_vec[name] = p.grad.reshape(-1) - self.prev_grad[name] 


        """ Update local_state """
        self.local_state = OrderedDict()
        self.local_state["primal"] = copy.deepcopy(self.primal_state)
        self.local_state["dual"] = OrderedDict()
        self.local_state["penalty"] = OrderedDict()
        self.local_state["penalty"][self.id] = 0.0
        self.local_state["svec"] = copy.deepcopy(self.s_vec)
        self.local_state["yvec"] = copy.deepcopy(self.y_vec)
        self.local_state["grad"] = copy.deepcopy(self.grad)

        return self.local_state
 
