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


class ClientOptimPCA21(BaseClient):
    def __init__(
        self, id, weight, model, dataloader, cfg, outfile, test_dataloader, **kwargs
    ):
        super(ClientOptimPCA21, self).__init__(
            id, weight, model, dataloader, cfg, outfile, test_dataloader
        )
        self.__dict__.update(kwargs)

        self.loss_fn = eval(self.loss_type)

        self.round = 0

        ## construct
        self.P, self.EVR = super(ClientOptimPCA21, self).construct_projection_matrix()
        super(ClientOptimPCA21, self).log_pca()
        super(ClientOptimPCA21, self).client_log_title()

    def update(self, global_state_vec):

        """Inputs for the local model update"""
        if self.round == 0:
            pca_dir = self.pca_dir  + "/client_%s" % (self.id)
            # Resume from params_start
            self.model.load_state_dict(
                torch.load(
                    os.path.join(pca_dir, "0.pt"),
                    map_location=torch.device(self.cfg.device),
                )
            )
        else: 
            
            local_state_vec = torch.mm(self.P.transpose(0, 1), global_state_vec)
  
            super(ClientOptimPCA21, self).update_param(local_state_vec)
 
 

        self.model.to(self.cfg.device)

        optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)

        """ Multiple local update """
        start_time=time.time()
        for t in range(self.num_local_epochs):

            if self.test_dataloader != None:
                train_loss, train_accuracy = super(
                    ClientOptimPCA21, self
                ).client_validation(self.dataloader)
                test_loss, test_accuracy = super(
                    ClientOptimPCA21, self
                ).client_validation(self.test_dataloader)
                per_iter_time = time.time() - start_time
                super(ClientOptimPCA21, self).client_log_content(
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
                grad = super(ClientOptimPCA21, self).get_model_grad_vec()

                ## reduced gradient
                gk = torch.mm(self.P, grad.reshape(-1, 1))

                ## back to original space
                grad_proj = torch.mm(self.P.transpose(0, 1), gk)                    
                super(ClientOptimPCA21, self).update_grad(grad_proj)                

                optimizer.step()
 

        self.round += 1

        ## Reduction 
        param_vec = super(ClientOptimPCA21, self).get_model_param_vec()        
        param_vec = torch.tensor(param_vec, device = self.cfg.device)        
        param_vec = param_vec.reshape(-1, 1)        
        param_vec = torch.mm(self.P, param_vec) 
 
        # self.primal_state = copy.deepcopy(self.model.state_dict())
 
        """ Update local_state """
        self.local_state = OrderedDict()
        # self.local_state["primal"] = copy.deepcopy(self.primal_state)   
        self.local_state["param_vec"] = param_vec     
        self.local_state["dual"] = OrderedDict()
        self.local_state["penalty"] = OrderedDict()
        self.local_state["penalty"][self.id] = 0.0        

        return self.local_state
