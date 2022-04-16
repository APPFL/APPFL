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


class ClientOptimPCA(BaseClient):
    def __init__(
        self, id, weight, model, dataloader, cfg, outfile, test_dataloader, **kwargs
    ):
        super(ClientOptimPCA, self).__init__(
            id, weight, model, dataloader, cfg, outfile, test_dataloader
        )
        self.__dict__.update(kwargs)

        self.loss_fn = eval(self.loss_type)

        self.round = 0

        ## construct
        self.P, self.EVR = super(ClientOptimPCA, self).construct_projection_matrix()
        super(ClientOptimPCA, self).log_pca()
        super(ClientOptimPCA, self).client_log_title()
        
        

    def update(self):

        """Inputs for the local model update"""

        self.model.to(self.cfg.device)

        optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)

        """ Multiple local update """
        for t in range(self.num_local_epochs):

            if self.cfg.validation == True and self.test_dataloader != None:
                train_loss, train_accuracy = super(ClientOptimPCA, self).client_validation(
                    self.dataloader
                )
                test_loss, test_accuracy = super(ClientOptimPCA, self).client_validation(
                    self.test_dataloader
                )
                super(ClientOptimPCA, self).client_log_content(
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
                    grad = super(ClientOptimPCA, self).get_model_grad_vec()  
            
                    ## reduced gradient
                    gk = torch.mm(self.P, grad.reshape(-1,1))
                    
                    ## back to original space
                    grad_proj = torch.mm(self.P.transpose(0, 1), gk)                

                    super(ClientOptimPCA, self).update_grad(grad_proj)


                optimizer.step()

                if self.clip_value != False:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.clip_value,
                        norm_type=self.clip_norm,
                    )

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

        """ Differential Privacy  """
        if self.epsilon != False:
            sensitivity = 0
            if self.clip_value != False:
                sensitivity = 2.0 * self.clip_value * self.optim_args.lr
            scale_value = sensitivity / self.epsilon
            super(ClientOptimPCA, self).laplace_mechanism_output_perturb(scale_value)

        """ Update local_state """
        self.local_state = OrderedDict()
        self.local_state["primal"] = copy.deepcopy(self.primal_state)
        self.local_state["dual"] = OrderedDict()
        self.local_state["penalty"] = OrderedDict()
        self.local_state["penalty"][self.id] = 0.0

        return self.local_state
