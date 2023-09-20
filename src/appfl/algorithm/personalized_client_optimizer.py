import logging
import os
from collections import OrderedDict
from .algorithm import BaseClient

import torch
from torch.optim import *

from ..misc.utils import save_model_state_iteration

from torch.utils.data import DataLoader
import copy

import numpy as np
import time


class PersonalizedClientOptim(BaseClient):
    
    def __init__(
        self, id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader, metric, **kwargs
    ):
        super(PersonalizedClientOptim, self).__init__(
            id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader
        )
        self.__dict__.update(kwargs)

        self.round = 0
        self.metric = metric

        super(PersonalizedClientOptim, self).client_log_title()

    def update(self):
        
        """Inputs for the local model update"""

        self.model.to(self.cfg.device)

        optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)
 

        """ Multiple local update """
        start_time=time.time()
        ## initial evaluation
        if self.cfg.validation == True and self.test_dataloader != None:
            test_loss, test_accuracy = super(PersonalizedClientOptim, self).client_validation(
                self.test_dataloader, self.metric
            )
            per_iter_time = time.time() - start_time
            super(PersonalizedClientOptim, self).client_log_content(
                0, per_iter_time, 0, 0, test_loss, test_accuracy
            )
            ## return to train mode
            self.model.train()        

        ## local training 
        for t in range(self.num_local_epochs):
            start_time=time.time()
            train_loss = 0
            train_accuracy = 0          
            y_true = []
            y_pred = []
            for data, target in self.dataloader:   
                
                data = data.to(self.cfg.device)
                target = target.to(self.cfg.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                y_true.append(target.detach().cpu().numpy())
                y_pred.append(output.detach().cpu().numpy())

                if self.clip_value != False:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.clip_value,
                        norm_type=self.clip_norm,
                    )
            ## Validation
            train_loss = train_loss / len(self.dataloader)
            y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
            train_accuracy = float(self.metric(y_true,y_pred))
            
            if self.cfg.validation == True and self.test_dataloader != None:
                test_loss, test_accuracy = super(PersonalizedClientOptim, self).client_validation(
                    self.test_dataloader, self.metric
                )
                per_iter_time = time.time() - start_time
                super(PersonalizedClientOptim, self).client_log_content(
                    t+1, per_iter_time, train_loss, train_accuracy, test_loss, test_accuracy
                )
                ## return to train mode
                self.model.train()
 
        self.round += 1

        self.primal_state = copy.deepcopy(self.model.state_dict())
        if (self.cfg.device == "cuda"):            
            for k in self.primal_state:
                self.primal_state[k] = self.primal_state[k].cpu()

        """ Differential Privacy  """
        if self.epsilon != False:
            sensitivity = 0
            if self.clip_value != False:
                sensitivity = 2.0 * self.clip_value * self.optim_args.lr
            scale_value = sensitivity / self.epsilon
            super(PersonalizedClientOptim, self).laplace_mechanism_output_perturb(scale_value)
            
        """ Save each client model periodically """ 
        if self.cfg.personalization == True and self.cfg.save_model_state_dict == True and ((self.round) % self.cfg.checkpoints_interval == 0 or self.round== self.cfg.num_epochs):
            save_model_state_iteration(self.round, self.model, self.cfg, client_id=self.id)

        """ Update local_state """
        self.local_state = OrderedDict()
        self.local_state["primal"] = self.primal_state
        self.local_state["dual"] = OrderedDict()
        self.local_state["penalty"] = OrderedDict()
        self.local_state["penalty"][self.id] = 0.0

        return self.local_state
 
