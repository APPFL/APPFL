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


class ClientOptimPSGD(BaseClient):
    def __init__(
        self, id, weight, model, dataloader, cfg, outfile, test_dataloader, **kwargs
    ):
        super(ClientOptimPSGD, self).__init__(
            id, weight, model, dataloader, cfg, outfile, test_dataloader
        )
        self.__dict__.update(kwargs)

        self.loss_fn = eval(self.loss_type)

        self.round = 0 

        ## construct
        self.P, self.EVR = super(ClientOptimPSGD, self).construct_projection_matrix()        
        super(ClientOptimPSGD, self).client_log_title()

    def update(self):


        self.model.to(self.cfg.device)

        optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)

        """ Multiple local update """
        start_time=time.time()
        reduced_grad = 0
        for t in range(self.num_local_epochs):

            if self.test_dataloader != None:
                train_loss, train_accuracy = super(
                    ClientOptimPSGD, self
                ).client_validation(self.dataloader)
                test_loss, test_accuracy = super(
                    ClientOptimPSGD, self
                ).client_validation(self.test_dataloader)
                per_iter_time = time.time() - start_time
                super(ClientOptimPSGD, self).client_log_content(
                    t, per_iter_time, train_loss, train_accuracy, test_loss, test_accuracy
                )
                ## return to train mode
                self.model.train()
            
            start_time=time.time()
            for data, target in self.dataloader:
                data = data.to(self.cfg.device)
                target = target.to(self.cfg.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()
 
                ## gradient
                grad = super(ClientOptimPSGD, self).get_model_grad_vec()

                ## reduced gradient
                gk = torch.mm(self.P, grad.reshape(-1, 1))
 
                reduced_grad += optimizer.param_groups[0]['lr'] * gk                

                ## back to original space
                grad_proj = torch.mm(self.P.transpose(0, 1), gk)                    
                super(ClientOptimPSGD, self).update_grad(grad_proj)                

                optimizer.step()
 
  

        if self.test_dataloader != None:
            train_loss, train_accuracy = super(
                ClientOptimPSGD, self
            ).client_validation(self.dataloader)
            test_loss, test_accuracy = super(
                ClientOptimPSGD, self
            ).client_validation(self.test_dataloader)
            per_iter_time = time.time() - start_time
            super(ClientOptimPSGD, self).client_log_content(
                self.num_local_epochs, per_iter_time, train_loss, train_accuracy, test_loss, test_accuracy
            )
         

        self.round += 1


        """ Update local_state """
        self.local_state = OrderedDict()
        self.local_state["primal"] = OrderedDict()
        self.local_state["dual"] = OrderedDict()
        self.local_state["grad"] = reduced_grad     
        self.local_state["penalty"] = OrderedDict()
        self.local_state["penalty"][self.id] = 0.0 
         
        return self.local_state
