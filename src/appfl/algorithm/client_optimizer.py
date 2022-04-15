import logging
import os
from collections import OrderedDict
from .algorithm import BaseClient

import torch
from torch.optim import *
 
from torch.utils.data import DataLoader
import copy


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
 