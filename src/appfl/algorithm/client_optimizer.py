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
        self, id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader, **kwargs
    ):
        super(ClientOptim, self).__init__(
            id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader
        )
        self.__dict__.update(kwargs)

        self.round = 0

        super(ClientOptim, self).client_log_title()

    def update(self):
        
        """Inputs for the local model update"""

        self.model.to(self.cfg.device)

        optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)

        """ Multiple local update """
        start_time=time.time()
        ## initial evaluation
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

        ## local training 
        for t in range(self.num_local_epochs):
            start_time=time.time()
            train_loss = 0
            train_correct = 0            
            tmptotal = 0
            for data, target in self.dataloader:                
                tmptotal += len(target)
                data = data.to(self.cfg.device)
                target = target.unsqueeze(1).to(self.cfg.device)
                optimizer.zero_grad()
                output = self.model(data)
                
                target = target.type_as(output)
                probs = torch.sigmoid(output)
                pred = probs > 0.5
                
                loss = self.loss_fn(probs, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                # if output.shape[1] == 1:
                #     pred = torch.round(output)
                # else:
                #     pred = output.argmax(dim=1, keepdim=True)
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
                test_loss, test_accuracy = super(ClientOptim, self).client_validation(
                    self.test_dataloader
                )
                per_iter_time = time.time() - start_time
                super(ClientOptim, self).client_log_content(
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
        if (self.cfg.device == "cuda"):            
            for k in self.primal_state:
                self.primal_state[k] = self.primal_state[k].cpu()

        """ Differential Privacy  """
        if self.epsilon != False:
            sensitivity = 0
            if self.clip_value != False:
                sensitivity = 2.0 * self.clip_value * self.optim_args.lr
            scale_value = sensitivity / self.epsilon
            super(ClientOptim, self).laplace_mechanism_output_perturb(scale_value)

        """ Update local_state """
        self.local_state = OrderedDict()
        self.local_state["primal"] = self.primal_state
        self.local_state["dual"] = OrderedDict()
        self.local_state["penalty"] = OrderedDict()
        self.local_state["penalty"][self.id] = 0.0

        return self.local_state
 
class FedMTLClient(BaseClient):
    def __init__(self, id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader, **kwargs):
        super(FedMTLClient, self).__init__(
            id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader
        )
        self.__dict__.update(kwargs)

        self.round = 0

        super(FedMTLClient, self).client_log_title()
    
    def client_validation_MTL(self, dataloader):
        
        # print('ewotuwt')

        if self.loss_fn is None or dataloader is None:
            return 0.0, 0.0

        self.model.to(self.cfg.device)
        self.model.eval()
        loss = 0
        correct = 0
        tmpcnt = 0
        tmptotal = 0
        with torch.no_grad():
            for sample in self.dataloader:          
                tmpcnt += 1      
                data = sample['img'].to(self.cfg.device)
                targets = sample['targets']
                target = targets[0].unsqueeze(1).to(self.cfg.device)
                tmptotal += len(target)               
                preds_all = self.model(data)
                output = preds_all[0]
                
                target = target.type_as(output)
                probs = torch.sigmoid(output)
                pred = probs > 0.5
                
                loss += self.loss_fn[0](probs, target).item()

                # if output.shape[1] == 1:
                #     pred = torch.round(output)
                # else:
                #     pred = output.argmax(dim=1, keepdim=True)

                correct += pred.eq(target.view_as(pred)).sum().item()

        # FIXME: do we need to sent the model to cpu again?
        # self.model.to("cpu")

        loss = loss / tmpcnt
        accuracy = 100.0 * correct / tmptotal

        return loss, accuracy
        
    # update with multiple losses
    def update(self):
        """Inputs for the local model update"""

        self.model.to(self.cfg.device)

        optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)
 

        """ Multiple local update """
        start_time=time.time()
        ## initial evaluation
        if self.cfg.validation == True and self.test_dataloader != None:
            test_loss, test_accuracy = self.client_validation_MTL(
                self.test_dataloader
            )
            per_iter_time = time.time() - start_time
            super(FedMTLClient, self).client_log_content(
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
            for sample in self.dataloader:                
                data = sample['img'].to(self.cfg.device)
                targets = sample['targets']
                
                for i in range(len(targets)):
                    targets[i] = targets[i].to(self.cfg.device)
                labels = targets[0].unsqueeze(1)
                tmptotal += len(labels)
                
                optimizer.zero_grad()
                
                preds_all = self.model(data)
                output = preds_all[0]
                
                labels = labels.type_as(output)
                probs = torch.sigmoid(output)
                pred = probs > 0.5
                                   
                
                loss = self.loss_fn[0](probs, labels)
                if self.id != self.cfg.fed.args.target:
                    for idx, c in enumerate(self.loss_fn[1:]):
                        loss += c(preds_all[idx+1], targets[idx+1])
                # else:
                    # loss = self.loss_fn[0](output, labels)

                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                # if output.shape[1] == 1:
                #     pred = torch.round(output)
                # else:
                #     pred = output.argmax(dim=1, keepdim=True)
                train_correct += pred.eq(labels.view_as(pred)).sum().item()

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
                test_loss, test_accuracy = self.client_validation_MTL(
                    self.test_dataloader
                )
                per_iter_time = time.time() - start_time
                super(FedMTLClient, self).client_log_content(
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
        if (self.cfg.device == "cuda"):            
            for k in self.primal_state:
                self.primal_state[k] = self.primal_state[k].cpu()

        """ Differential Privacy  """
        if self.epsilon != False:
            sensitivity = 0
            if self.clip_value != False:
                sensitivity = 2.0 * self.clip_value * self.optim_args.lr
            scale_value = sensitivity / self.epsilon
            super(FedMTLClient, self).laplace_mechanism_output_perturb(scale_value)

        """ Update local_state """
        self.local_state = OrderedDict()
        self.local_state["primal"] = self.primal_state
        self.local_state["dual"] = OrderedDict()
        self.local_state["penalty"] = OrderedDict()
        self.local_state["penalty"][self.id] = 0.0

        return self.local_state