import logging

from collections import OrderedDict
from .algorithm import BaseServer, BaseClient

import torch
from torch.optim import *
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import copy
import numpy as np

class FedServerBFGS(BaseServer):
    def __init__(self, weights, model, num_clients, device, **kwargs):
        super(FedServerBFGS, self).__init__(weights, model, num_clients, device)
        self.__dict__.update(kwargs)
        self.logger = logging.getLogger(__name__)


        self.global_state_vec = 0        
        self.pseudo_grad = 0
        self.m_vector = 0         
        self.v_vector = 0

        ## construct projection        
        self.P, self.EVR = super(FedServerBFGS, self).construct_projection_matrix()            

        ## approximate inverse Hessian matrix
        self.H_matrix = torch.eye(self.ncomponents, device=self.device)
        self.I_matrix = torch.eye(self.ncomponents, device=self.device)


        ##
        self.round = 0
        self.step_prev = 0
        self.pseudo_grad_prev = 0
        self.global_state_vec_prev = 0
        

    def compute_pseudo_gradient(self):

        self.pseudo_grad = 0
               
        for id in range(self.num_clients):             
            self.pseudo_grad += self.weights[id] * self.reduced_grad_vec[id]

    def update_m_vector(self):

        self.m_vector = self.server_momentum_param_1 * self.m_vector + (1.0 - self.server_momentum_param_1) * self.pseudo_grad
  

    def update(self, local_states: OrderedDict, clients):

        """Inputs for the global model update""" 

        global_state_vec = super(FedServerBFGS, self).get_model_param_vec()        
        global_state_vec = torch.tensor(global_state_vec, device = self.device)
        self.global_state_vec = global_state_vec.reshape(-1,1)
 
        super(FedServerBFGS, self).grad_vec_recover_from_local_states(local_states)
        
        self.update_global_state(clients)
        
        super(FedServerBFGS, self).update_param(self.global_state_vec)

        
    def backtracking_line_search(self, clients):
        ## deepcopy the models        
        model = {}                
        for k, client in enumerate(clients):                        
            model[k] = copy.deepcopy(client.model)             
            
        ## update model     
        self.update_model_param(model, self.global_state_vec)            
        
        ## compute loss_prev        
        loss_prev = 0
        for k, client in enumerate(clients):            
            loss_prev += self.loss_calculation(client.loss_type, model[k], client.dataloader)
        loss_prev = loss_prev / self.num_clients
        
        termination = 1
        while termination:
            
            ##  
            RHS = loss_prev + self.c * step_size * torch.dot(self.pseudo_grad.reshape(-1), direction.reshape(-1))

            ##
            global_state_vec_next = self.global_state_vec + torch.mm( self.P.transpose(0, 1), step_size * direction )


            self.update_model_param(model, global_state_vec_next)            

            ## compute loss_new        
            loss_new = 0
            for k, client in enumerate(clients):            
                loss_new += self.loss_calculation(client.loss_type, model[k], client.dataloader)
            loss_new = loss_new / self.num_clients

            if loss_new <= RHS or step_size <= 1e-10:
                termination = 0
            else:
                step_size = step_size * self.tau

        return step_size

    def update_model_param(self, model, vector):
        for k in range(self.num_clients):
            idx = 0
            for _,param in model[k].named_parameters():
                arr_shape = param.data.shape
                size = 1
                for i in range(len(list(arr_shape))):
                    size *= arr_shape[i]
                param.data = vector[idx:idx+size].reshape(arr_shape)
                idx += size  


    def loss_calculation(self, loss_type, model, dataloader):
        device = self.device
 
        loss_fn = eval(loss_type)
         
        model.to(device)
        model.eval()
        loss = 0
        tmpcnt = 0 
        with torch.no_grad():
            for img, target in dataloader:
                tmpcnt += 1 
                img = img.to(device)
                target = target.to(device)
                output = model(img)
                loss +=  loss_fn(output, target).item()

        loss = loss / tmpcnt        

        return loss          

    def logging_iteration(self, cfg, logger, t):
        if t == 0:
            title = super(FedServerBFGS, self).log_title()
            logger.info(title)

        contents = super(FedServerBFGS, self).log_contents(cfg, t)
        logger.info(contents)

    def logging_summary(self, cfg, logger):
        super(FedServerBFGS, self).log_summary(cfg, logger)
