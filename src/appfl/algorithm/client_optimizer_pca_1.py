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


class ClientOptimPCA1(BaseClient):
    def __init__(
        self, id, weight, model, dataloader, cfg, outfile, test_dataloader, **kwargs
    ):
        super(ClientOptimPCA1, self).__init__(
            id, weight, model, dataloader, cfg, outfile, test_dataloader
        )
        self.__dict__.update(kwargs)

        self.loss_fn = eval(self.loss_type)

        self.round = 0

        ## construct
        self.P, self.EVR = super(ClientOptimPCA1, self).construct_projection_matrix()
        super(ClientOptimPCA1, self).log_pca()
        super(ClientOptimPCA1, self).client_log_title()

    def update(self):

        """Inputs for the local model update"""
        if self.round == 0:
            pca_dir = self.cfg.pca_dir + "_%s" % (self.id)
            # Resume from params_start
            self.model.load_state_dict(
                torch.load(
                    os.path.join(pca_dir, "0.pt"),
                    map_location=torch.device(self.cfg.device),
                )
            )
        
        


        self.model.to(self.cfg.device)

        optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)

        """ Multiple local update """
        for t in range(self.num_local_epochs):

            if self.cfg.validation == True and self.test_dataloader != None:
                train_loss, train_accuracy = super(
                    ClientOptimPCA1, self
                ).client_validation(self.dataloader)
                test_loss, test_accuracy = super(
                    ClientOptimPCA1, self
                ).client_validation(self.test_dataloader)
                super(ClientOptimPCA1, self).client_log_content(
                    t, train_loss, train_accuracy, test_loss, test_accuracy
                )
                ## return to train mode
                self.model.train()

            for data, target in self.dataloader:
                data = data.to(self.cfg.device)
                target = target.to(self.cfg.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()

                if self.cfg.projection:
                    ## gradient
                    grad = super(ClientOptimPCA1, self).get_model_grad_vec()

                    ## reduced gradient
                    gk = torch.mm(self.P, grad.reshape(-1, 1))

                    ## back to original space
                    grad_proj = torch.mm(self.P.transpose(0, 1), gk)

                    super(ClientOptimPCA1, self).update_grad(grad_proj)

                optimizer.step()
 
 

        self.round += 1

        # ## Reduction 
        # param_vec = super(ClientOptimPCA, self).get_model_param_vec()
        # print("client_param_vec=", param_vec.shape)
        # param_vec = torch.tensor(param_vec, device = self.cfg.device)        
        # param_vec = param_vec.reshape(-1, 1)        
        # param_vec = torch.mm(self.P, param_vec) 
        # print("client_param_vec_reduced=", param_vec.shape)
 

        """ Update local_state """
        self.local_state = OrderedDict()
        self.local_state["primal"] = OrderedDict()
        self.local_state["dual"] = OrderedDict()
        self.local_state["penalty"] = OrderedDict()
        self.local_state["penalty"][self.id] = 0.0
        # self.local_state["param_vec"] = param_vec

        return self.local_state
