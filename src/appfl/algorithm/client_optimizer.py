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
    def __init__(
        self,
        id,
        weight,
        model,
        loss_fn,
        dataloader,
        cfg,
        outfile,
        test_dataloader,
        **kwargs
    ):
        super(ClientOptim, self).__init__(
            id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader
        )
        self.__dict__.update(kwargs)

        self.round = 0

        super(ClientOptim, self).client_log_title()
    
    def training(self):
        """
        This function trains the model using "optimizer" such as SGD, Adam, and so on
        """
        optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)
        

        train_loss = 0
        train_correct = 0
        tmptotal = 0

        for data, target in self.dataloader:

            tmptotal += len(target)

            data, target = data.to(self.cfg.device), target.to(self.cfg.device)

            output = self.model(data)

            train_correct = self.counting_correct(output, target, train_correct)

            loss = self.loss_fn(output, target)

            train_loss += loss.item()

            optimizer.zero_grad()

            loss.backward()

            self.clipping_gradient()  ## works only when "self.clip_value != False"

            optimizer.step()

        train_loss = train_loss / len(self.dataloader)

        train_accuracy = 100.0 * train_correct / tmptotal

        return train_loss, train_accuracy

    def clipping_gradient(self):
        """
        This function clips the current gradient such that the "self.clip_norm" of the gradient is less than or equal to "self.clip_value"
        """
        if self.clip_value != False:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.clip_value,
                norm_type=self.clip_norm,
            )

    def counting_correct(self, output, target, train_correct):
        """
        This function evaluates how many correct labels are predicted using the current models
        """
        if self.loss_fn == "CrossEntropyLoss()":
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()

        elif self.loss_fn == "BCELoss()" or self.loss_fn == "BCEWithLogitsLoss()":
            pred = torch.round(output)
            train_correct += pred.eq(target.view_as(pred)).sum().item()

        else:
            train_correct = -1

        return train_correct

    def update(self):

        """Inputs for the local model update"""

        self.model.to(self.cfg.device)

        ## initial evaluation
        start_time = time.time()
        if self.cfg.validation == True and self.test_dataloader != None:
            test_loss, test_accuracy = super(ClientOptim, self).client_validation(
                self.test_dataloader
            )
            per_iter_time = time.time() - start_time
            super(ClientOptim, self).client_log_content(
                0, per_iter_time, 0, 0, test_loss, test_accuracy
            )
            ## return to train mode
            self.model.train()

        """ Multiple local update """
        for t in range(self.num_local_epochs):
            start_time = time.time()
            
            train_loss, train_accuracy = self.training()

            ## validation with test dataset
            if self.cfg.validation == True and self.test_dataloader != None:
                test_loss, test_accuracy = super(ClientOptim, self).client_validation(
                    self.test_dataloader
                )
                per_iter_time = time.time() - start_time
                super(ClientOptim, self).client_log_content(
                    t + 1,
                    per_iter_time,
                    train_loss,
                    train_accuracy,
                    test_loss,
                    test_accuracy,
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

        """ Differential Privacy  """
        if self.epsilon != False:
            sensitivity = 0
            if self.clip_value != False:
                sensitivity = 2.0 * self.clip_value * self.optim_args.lr
            scale_value = sensitivity / self.epsilon
            super(ClientOptim, self).laplace_mechanism_output_perturb(scale_value)

        """ Update local_state """
        self.local_state = OrderedDict()
        self.local_state["primal"] = copy.deepcopy(self.primal_state)
        self.local_state["dual"] = OrderedDict()
        self.local_state["penalty"] = OrderedDict()
        self.local_state["penalty"][self.id] = 0.0

        return self.local_state



