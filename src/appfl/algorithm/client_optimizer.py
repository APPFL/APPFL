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


class ClientOptim(BaseClient):
    """This client optimizer which perform updates for certain number of epochs in each training round."""
    def __init__(
        self, id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader, metric, **kwargs
    ):
        super(ClientOptim, self).__init__(id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader)
        self.__dict__.update(kwargs)
        self.metric = metric
        super(ClientOptim, self).client_log_title()

    def update(self):
        self.model.to(self.cfg.device)
        optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)
        """ Multiple local update """
        ## Initial evaluation
        if self.cfg.validation == True and self.test_dataloader != None:
            start_time=time.time()
            test_loss, test_accuracy = super(ClientOptim, self).client_validation(self.test_dataloader, self.metric)
            per_iter_time = time.time() - start_time
            super(ClientOptim, self).client_log_content(0, per_iter_time, 0, 0, test_loss, test_accuracy)
            ## Return to train mode
            self.model.train()        

        ## Local training 
        for t in range(self.num_local_epochs):
            start_time=time.time()
            for data, target in self.dataloader:                
                data = data.to(self.cfg.device)
                target = target.to(self.cfg.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()
                optimizer.step()
                if self.clip_value != False:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.clip_value,
                        norm_type=self.clip_norm,
                    )
            ## Validation
            if self.cfg.validation == True and self.test_dataloader != None:
                train_loss, train_accuracy = super(ClientOptim, self).client_validation(self.dataloader, self.metric)
                test_loss, test_accuracy = super(ClientOptim, self).client_validation(self.test_dataloader, self.metric)
                per_iter_time = time.time() - start_time
                super(ClientOptim, self).client_log_content(t+1, per_iter_time, train_loss, train_accuracy, test_loss, test_accuracy)
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
        if (self.cfg.device == "cuda"):            
            for k in self.primal_state:
                self.primal_state[k] = self.primal_state[k].cpu()

        ## Differential Privacy
        if self.epsilon != False:
            sensitivity = 0
            if self.clip_value != False:
                sensitivity = 2.0 * self.clip_value * self.optim_args.lr
            scale_value = sensitivity / self.epsilon
            super(ClientOptim, self).laplace_mechanism_output_perturb(scale_value)

        ## Update local_state
        self.local_state = OrderedDict()
        self.local_state["primal"] = self.primal_state
        self.local_state["dual"] = OrderedDict()
        self.local_state["penalty"] = OrderedDict()
        self.local_state["penalty"][self.id] = 0.0

        return self.local_state
 