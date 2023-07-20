import logging
from collections import OrderedDict

from .server_federated import FedServer
from .algorithm import BaseClient

import torch
from torch.optim import *
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import copy
from .client_optimizer import ClientOptim
import numpy as np
import time
import os

class ServerFedGP(FedServer):
    def __init__(self, weights, model, loss_fn, num_clients, beta, clients_size, device, **kwargs):
        super(ServerFedGP, self).__init__(weights, model, loss_fn, num_clients, device, **kwargs)
        # initialization
        self.beta = beta # bias-variance trade-off factor
        # self.target  # target idx to know how to aggregate the gradients
        self.source_grads = OrderedDict()
        self.target_grad = OrderedDict()
        self.clients_size = clients_size

    def compute_step(self):
        super(ServerFedGP, self).compute_pseudo_gradient()
        for name in self.model.state_dict():
        # for name, _ in self.model.named_parameters():
            self.step[name] = -self.pseudo_grad[name]
    
    def compute_source_target_gradients(self):
        for idx in range(self.num_clients):
            if idx == self.target:
                for name in self.model.state_dict():
                    self.target_grad[name] = torch.zeros_like(self.model.state_dict()[name])
                    self.target_grad[name] = self.primal_states[idx][name] - self.global_state[name] 
            else:
                source_grad = OrderedDict()
                for name in self.model.state_dict():
                    source_grad[name] = torch.zeros_like(self.model.state_dict()[name])
                    source_grad[name] = self.primal_states[idx][name] - self.global_state[name] 
                self.source_grads[idx] = source_grad
    
    def update(self, local_states: OrderedDict, round_id):
        """Inputs for the global model update"""
        self.global_state = copy.deepcopy(self.model.state_dict())
        super(FedServer, self).primal_recover_from_local_states(local_states)

        """ residual calculation """
        super(FedServer, self).primal_residual_at_server()
 
        """ change device """
        for i in range(self.num_clients): 
            for name in self.model.state_dict():
                self.primal_states[i][name] = self.primal_states[i][name].to(
                    self.device
                )
        
        if round_id < 2:
            # go fedavg
            self.update_fedavg()
        
        else:
            # go fedgp
            self.update_fedgp()
    
    def update_fedavg(self):
        """ global_state calculation """
        self.compute_step() 
        for name in self.model.state_dict():   
            self.global_state[name] = (self.global_state[name]).float()     
            if name in self.list_named_parameters: 
                self.global_state[name] += self.step[name]            
            else:
                tmpsum = torch.zeros_like(self.global_state[name], device=self.device)                
                for i in range(self.num_clients):
                    tmpsum += self.primal_states[i][name]                
                self.global_state[name] = torch.div(tmpsum, self.num_clients)
                

        """ model update """
        self.model.load_state_dict(self.global_state)
    
    # update global model using the consine projection from target toward source directions
    def update_fedgp(self):
        self.compute_source_target_gradients()
        # ret_dict = copy.deepcopy(old_global_model_dict)
        b = self.beta
        cos = torch.nn.CosineSimilarity()
        for name in self.global_state:        
            # if name in self.list_named_parameters: 
            # if self.global_state[name].shape != torch.Size([]):
                # self.global_state[name] += self.step[name]
        # for key in ret_dict.keys():
            if self.global_state[name].shape != torch.Size([]):
                target_grad = self.target_grad[name] # target persudo gradient
                for idx in self.source_grads:
                    local_grad = self.source_grads[idx][name] # the rest of the persudo gradients
                    cur_sim = cos(target_grad.reshape(1,-1).float(), local_grad.reshape(1,-1).float())
                    if cur_sim > 0:
                        self.global_state[name] += b * self.target_lr_ratio * ((self.n_target_samples/self.target_batch_size)/(self.clients_size[idx]/self.source_batch_size)) * self.weights[idx] * cur_sim * local_grad
                self.global_state[name] += (1-b) * target_grad
            # else:
            #     tmpsum = torch.zeros_like(self.global_state[name], device=self.device)                
            #     for i in range(self.num_clients):
            #         tmpsum += self.primal_states[i][name]                
            #     self.global_state[name] = torch.div(tmpsum, self.num_clients)
        #         ret_dict[key] = old_global_model_dict[key]
        # return ret_dict
        
        """ model update """
        self.model.load_state_dict(self.global_state)
        
        
    def logging_summary(self, cfg, logger):
        super(FedServer, self).log_summary(cfg, logger)

        logger.info("client_learning_rate = %s " % (cfg.fed.args.optim_args.lr))

        if cfg.summary_file != "":
            with open(cfg.summary_file, "a") as f:

                f.write(
                    cfg.logginginfo.DataSet_name
                    + " Federated Gradient Projection ClientLR "
                    + str(cfg.fed.args.optim_args.lr)
                    + " TestAccuracy "
                    + str(cfg.logginginfo.accuracy)
                    + " BestAccuracy "
                    + str(cfg.logginginfo.BestAccuracy)
                    + " Time "
                    + str(round(cfg.logginginfo.Elapsed_time, 2))
                    + "\n"
                )