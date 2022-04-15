import logging
import os
from collections import OrderedDict
from .algorithm import BaseClient

import torch
from torch.optim import *
 
from torch.utils.data import DataLoader
import copy

import numpy as np

class ClientOptim(BaseClient):
    def __init__(self, cfg, id, weight, model, dataloader, device, **kwargs):
        super(ClientOptim, self).__init__(cfg, id, weight, model, dataloader, device)
        self.__dict__.update(kwargs)

        self.loss_fn = eval(self.loss_type)

        self.round = 0
        
           
    def update(self, outfile, outdir, test_dataloader: DataLoader=None ):
  
        """ Inputs for the local model update """        

        self.model.train()
        self.model.to(self.device)

        optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)


        """ Multiple local update """
        for t in range(self.num_local_epochs):

            if self.cfg.validation == True:            
                train_loss, train_accuracy = super(ClientOptim, self).validation_client(copy.deepcopy(self.model), self.dataloader)
                test_loss, test_accuracy = super(ClientOptim, self).validation_client(copy.deepcopy(self.model), test_dataloader)
                outfile = super(ClientOptim, self).write_result_content(outfile, t, train_loss, train_accuracy, test_loss, test_accuracy)
  
            for data, target in self.dataloader:
                data = data.to(self.device)
                target = target.to(self.device)
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

            ## save model.state_dict()
            if self.cfg.save_model_state_dict == True:
                torch.save(self.model.state_dict(), os.path.join(outdir,  str(t) +  '.pt'))

 
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

        return self.local_state, outfile



    ############################################################
    #### PSGD
    ############################################################

    def write_result(self, output_filename, W, P, explained_variance_ratio_): 

        dir = self.cfg.output_dirname  
        if os.path.isdir(dir) == False:
            os.mkdir(dir)
         
        file_ext = ".txt"
        filename = dir + "/%s%s" % (output_filename, file_ext)
        uniq = 1
        while os.path.exists(filename):
            filename = dir + "/%s_%d%s" % (output_filename, uniq, file_ext)
            uniq += 1
        
        outfile = open(filename, "w")

        outfile.write("W: (%s, %s) \n" %(W.shape[0], W.shape[1]))
        outfile.write("Ratio: %s \n" %(explained_variance_ratio_))
        outfile.write("Sum: %s \n" %(sum(explained_variance_ratio_)))
        outfile.write("P: (%s, %s) \n" %(P.shape[0], P.shape[1]))

        title = "%10s %10s %10s %10s %10s %10s \n" % (
            "Round",
            "LocalEpoch", 
            "TrainLoss",
            "TrainAccu",
            "TestLoss",
            "TestAccu",
        ) 
        outfile.write(title)
        
        return outfile, dir    

    def get_model_param_vec(self, model):
        """
        Return model parameters as a vector
        """
        vec = []
        for _,param in model.named_parameters():
            vec.append(param.detach().cpu().numpy().reshape(-1))
        return np.concatenate(vec, 0)

    def get_model_grad_vec(self, model):
        # Return the model grad as a vector

        vec = []
        for _,param in model.named_parameters():
            vec.append(param.grad.detach().reshape(-1))
        return torch.cat(vec, 0)

    def update_grad(self, model, grad_vec):
        idx = 0
        for _,param in model.named_parameters():
            arr_shape = param.grad.shape
            size = 1
            for i in range(len(list(arr_shape))):
                size *= arr_shape[i]
            param.grad.data = grad_vec[idx:idx+size].reshape(arr_shape)
            idx += size

    def update_param(self, model, param_vec):
        idx = 0
        for _,param in model.named_parameters():
            arr_shape = param.data.shape
            size = 1
            for i in range(len(list(arr_shape))):
                size *= arr_shape[i]
            param.data = param_vec[idx:idx+size].reshape(arr_shape)
            idx += size    


    def psgd_update(self, P, outfile, outdir, test_dataloader: DataLoader=None ):
  
        """ Inputs for the local model update """        

        self.model.train()
        self.model.to(self.device)

        optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)


        """ Multiple local update """
        for t in range(self.num_local_epochs):

            if self.cfg.validation == True:            
                train_loss, train_accuracy = super(ClientOptim, self).validation_client(copy.deepcopy(self.model), self.dataloader)
                test_loss, test_accuracy = super(ClientOptim, self).validation_client(copy.deepcopy(self.model), test_dataloader)
                outfile = super(ClientOptim, self).write_result_content(outfile, t, train_loss, train_accuracy, test_loss, test_accuracy)
  
            for data, target in self.dataloader:
                data = data.to(self.device)
                target = target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()

                ## gradient
                grad = self.get_model_grad_vec(self.model)                     
                ## reduced gradient
                gk = torch.mm(P, grad.reshape(-1,1))
                ## back to original space
                grad_proj = torch.mm(P.transpose(0, 1), gk)                
                self.update_grad(self.model, grad_proj)

                optimizer.step()

                if self.clip_value != False:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.clip_value,
                        norm_type=self.clip_norm,
                    )

            ## save model.state_dict()
            if self.cfg.save_model_state_dict == True:
                torch.save(self.model.state_dict(), os.path.join(outdir,  str(t) +  '.pt'))

 
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

        return self.local_state, outfile