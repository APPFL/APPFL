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

 


class ClientOptimPBFGS(BaseClient):
    def __init__(
        self, id, weight, model, dataloader, cfg, outfile, test_dataloader, **kwargs
    ):
        super(ClientOptimPBFGS, self).__init__(
            id, weight, model, dataloader, cfg, outfile, test_dataloader
        )
        self.__dict__.update(kwargs)

        self.loss_fn = eval(self.loss_type)

        self.round = 0 

        ## construct
        self.P, self.EVR = super(ClientOptimPBFGS, self).construct_projection_matrix()
        super(ClientOptimPBFGS, self).log_pca()
        super(ClientOptimPBFGS, self).client_log_title()

        ## settings for BFGS
        # Set the update period of basis variables (per iterations)
        self.T = 1000

        # Set the momentum parameters
        self.gamma = 0.9
        self.alpha = 0.0
        self.grad_res_momentum = 0

        # Store the last gradient on basis variables for P_plus_BFGS update
        self.gk_last = None

        # Variables for BFGS and backtracking line search
        self.rho = 0.55
        self.rho = 0.4
        self.sigma = 0.4
        self.Bk = torch.eye(40, device=self.cfg.device) 
        self.sk = None

        # Store the backtracking line search times
        self.search_times = []

    def update(self):


        self.model.to(self.cfg.device)

        optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)

        """ Multiple local update """
        start_time=time.time()
        reduced_grad = 0
        for t in range(self.num_local_epochs):

            if self.test_dataloader != None:
                train_loss, train_accuracy = super(
                    ClientOptimPBFGS, self
                ).client_validation(self.dataloader)
                test_loss, test_accuracy = super(
                    ClientOptimPBFGS, self
                ).client_validation(self.test_dataloader)
                per_iter_time = time.time() - start_time
                super(ClientOptimPBFGS, self).client_log_content(
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
                grad = super(ClientOptimPBFGS, self).get_model_grad_vec()
                
                gk = self.P_plus_BFGS(optimizer, grad, loss.item(), data, target)

                reduced_grad += gk
 
        if self.test_dataloader != None:
            train_loss, train_accuracy = super(
                ClientOptimPBFGS, self
            ).client_validation(self.dataloader)
            test_loss, test_accuracy = super(
                ClientOptimPBFGS, self
            ).client_validation(self.test_dataloader)
            per_iter_time = time.time() - start_time
            super(ClientOptimPBFGS, self).client_log_content(
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




    def P_plus_BFGS(self, optimizer, grad, oldf, X, y):
        # P_plus_BFGS algorithm

        # global rho, sigma, Bk, sk, gk_last, grad_res_momentum, gamma, alpha, search_times

        gk = torch.mm(self.P, grad.reshape(-1,1))

        grad_proj = torch.mm(self.P.transpose(0, 1), gk)
        grad_res = grad - grad_proj.reshape(-1)

        # Quasi-Newton update
        if self.gk_last is not None:
            yk = gk - self.gk_last
            g = (torch.mm(yk.transpose(0, 1), self.sk))[0, 0]
            if (g > 1e-20):
                pk = 1. / g
                t1 = torch.eye(40, device=self.cfg.device) - torch.mm(pk * yk, self.sk.transpose(0, 1))
                self.Bk = torch.mm(torch.mm(t1.transpose(0, 1), self.Bk), t1) + torch.mm(pk * self.sk, self.sk.transpose(0, 1))
        
        self.gk_last = gk
        dk = -torch.mm(self.Bk, gk)

        # Backtracking line search
        m = 0
        search_times_MAX = 20
        descent = torch.mm(gk.transpose(0, 1), dk)[0,0]

        # Copy the original parameters
        model_name = 'temporary.pt'
        torch.save(self.model.state_dict(), model_name)

        self.sk = dk
        while (m < search_times_MAX):
            super(ClientOptimPBFGS, self).update_grad(torch.mm(self.P.transpose(0, 1), -self.sk).reshape(-1))
            optimizer.step()
            yp = self.model(X)
            loss = torch.nn.CrossEntropyLoss()(yp,y)
            newf = loss.item()
            self.model.load_state_dict(torch.load(model_name))

            if (newf < oldf + self.sigma * descent):
                # print ('(', m, LA.cond(Bk), ')', end=' ')
                self.search_times.append(m)
                break

            m = m + 1
            descent *= self.rho
            self.sk *= self.rho
        
        # Cannot find proper lr
        # if m == search_times:
        #     sk *= 0

        # SGD + momentum for the remaining part of gradient
        self.grad_res_momentum = self.grad_res_momentum * self.gamma + grad_res

        # Update the model grad and do a step

        grad_proj = torch.mm(self.P.transpose(0, 1), -self.sk).reshape(-1) + self.grad_res_momentum * self.alpha


        super(ClientOptimPBFGS, self).update_grad(grad_proj)
        optimizer.step()

        return -self.sk