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


class ClientOptimPCA22(BaseClient):
    def __init__(
        self, id, weight, model, dataloader, cfg, outfile, test_dataloader, **kwargs
    ):
        super(ClientOptimPCA22, self).__init__(
            id, weight, model, dataloader, cfg, outfile, test_dataloader
        )
        self.__dict__.update(kwargs)

        self.loss_fn = eval(self.loss_type)

        self.round = 0

        ## construct
        self.P, self.EVR = super(ClientOptimPCA22, self).construct_projection_matrix()
        super(ClientOptimPCA22, self).log_pca()
        super(ClientOptimPCA22, self).client_log_title()

    def update(self, global_state_vec_red):

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

            local_state_vec = super(ClientOptimPCA22, self).get_model_param_vec()       
            local_state_vec = torch.tensor(local_state_vec, device = self.cfg.device)        
            local_state_vec_red = torch.mm(self.P, local_state_vec.reshape(-1, 1))

        else:
            local_state_vec_red = copy.deepcopy(global_state_vec_red)
            local_state_vec = torch.mm(self.P.transpose(0, 1), local_state_vec_red.reshape(-1, 1))
            super(ClientOptimPCA22, self).update_param(local_state_vec)


         
 

        self.model.to(self.cfg.device)

        optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)

        """ Multiple local update """
        start_time=time.time()
        for t in range(self.num_local_epochs):
 
            if self.test_dataloader != None:                

                train_loss, train_accuracy = super(
                    ClientOptimPCA22, self
                ).client_validation(self.dataloader)
                test_loss, test_accuracy = super(
                    ClientOptimPCA22, self
                ).client_validation(self.test_dataloader)
                per_iter_time = time.time() - start_time
                super(ClientOptimPCA22, self).client_log_content(
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
                grad = super(ClientOptimPCA22, self).get_model_grad_vec()

                ## reduced gradient
                gk = torch.mm(self.P, grad.reshape(-1, 1))
        
                # local_state_vec_red = local_state_vec_red - self.optim_args.lr * gk
                local_state_vec_red = local_state_vec_red - 100 * gk
 

                local_state_vec = torch.mm(self.P.transpose(0, 1), local_state_vec_red)


                print("-------1111")
                print("train_loss=", train_loss, " train_accuracy=", train_accuracy)
                print("test_loss=", test_loss, " test_accuracy=", test_accuracy)


                super(ClientOptimPCA22, self).update_param(local_state_vec)

                train_loss, train_accuracy = super(
                    ClientOptimPCA22, self
                ).client_validation(self.dataloader)
                test_loss, test_accuracy = super(
                    ClientOptimPCA22, self
                ).client_validation(self.test_dataloader)
                print("-------2222")
                print("train_loss=", train_loss, " train_accuracy=", train_accuracy)
                print("test_loss=", test_loss, " test_accuracy=", test_accuracy)
        

            stop

        self.round += 1
 
 
        # self.primal_state = copy.deepcopy(self.model.state_dict())
 
        """ Update local_state """
        self.local_state = OrderedDict()
        # self.local_state["primal"] = copy.deepcopy(self.primal_state)   
        self.local_state["param_vec"] = local_state_vec_red     
        self.local_state["dual"] = OrderedDict()
        self.local_state["penalty"] = OrderedDict()
        self.local_state["penalty"][self.id] = 0.0        

        return self.local_state
