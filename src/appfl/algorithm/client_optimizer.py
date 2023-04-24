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

    def training_closure(self):
        """
        This function trains the model using "optimizer" such as LBFGS which requires to reevaluate the function multiple times,
        """
        optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)

        train_loss = 0
        train_correct = 0
        tmptotal = 0

        for data, target in self.dataloader:

            tmptotal += len(target)

            data, target = data.to(self.cfg.device), target.to(self.cfg.device)

            def closure():
                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                output = self.model(data)

                # Compute loss
                loss = self.loss_fn(output, target)

                # Backward pass
                loss.backward()

                return loss

            optimizer.step(closure)

            loss = closure()
            train_loss += loss.data.item()
            output = self.model(data)
            train_correct = self.counting_correct(output, target, train_correct)

        train_loss = train_loss / len(self.dataloader)

        train_accuracy = 100.0 * train_correct / tmptotal

        return train_loss, train_accuracy

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
 
            optimizer.step()

        train_loss = train_loss / len(self.dataloader)

        train_accuracy = 100.0 * train_correct / tmptotal

        return train_loss, train_accuracy 

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

    def _gather_flat(self):
        views = []
        for p in self.model.parameters():
            if p.data is None:
                view = p.new(p.numel()).zero_()
            elif p.data.is_sparse:
                view = p.data.to_dense().view(-1)
            else:
                view = p.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

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
        
        ## store an initial model for differential privacy
        if self.dp != "none":            
            init_model_vec = self._gather_flat()
          



        """ Multiple local update """
        for t in range(self.num_local_epochs):
            start_time = time.time()

            ## training
            if self.cfg.fed.args.optim == "LBFGS":
                ## Some optimization algorithms such as Conjugate Gradient and LBFGS need to reevaluate the function multiple times, so you have to pass in a closure that allows them to recompute your model.
                train_loss, train_accuracy = self.training_closure()
            else:
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
        

        """ Differential Privacy  """
        if self.dp != "none": 
            # compute a step moved without DP
            curr_model_vec = self._gather_flat()
            step_vec = init_model_vec - curr_model_vec
            # generate a noise to be added for ensuring DP
            noise_vec = super(ClientOptim, self).output_perturbation(step_vec)                          
            # update model
            new_model_vec = init_model_vec - step_vec + noise_vec
            # TODO: new_model_vec to model 
            


      
        self.primal_state = copy.deepcopy(self.model.state_dict())  
        
        """ Update local_state """        
        self.local_state = OrderedDict()
        self.local_state["primal"] = copy.deepcopy(self.primal_state)
        self.local_state["dual"] = OrderedDict()
        self.local_state["penalty"] = OrderedDict()
        self.local_state["penalty"][self.id] = 0.0

        return self.local_state
