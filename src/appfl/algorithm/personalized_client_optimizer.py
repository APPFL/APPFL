import copy
import time
import torch
import numpy as np
from torch.optim import *
from .fl_base import BaseClient
from ..misc.utils import save_partial_model_iteration

class PersonalizedClientOptim(BaseClient):
    def __init__(
        self, id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader, metric, **kwargs
    ):
        super(PersonalizedClientOptim, self).__init__(id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader, metric)
        self.__dict__.update(kwargs)
        super(PersonalizedClientOptim, self).client_log_title()

    def update(self):
        self.model.to(self.cfg.device)
        optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)
 
        ## initial evaluation
        if self.cfg.validation == True and self.test_dataloader != None:
            start_time=time.time()
            test_loss, test_accuracy = super(PersonalizedClientOptim, self).client_validation()
            per_iter_time = time.time() - start_time
            super(PersonalizedClientOptim, self).client_log_content(0, per_iter_time, 0, 0, test_loss, test_accuracy)  

        ## local training 
        for t in range(self.num_local_epochs):
            start_time=time.time()
            train_loss, target_true, target_pred = 0, [], []
            for data, target in self.dataloader:   
                data = data.to(self.cfg.device)
                target = target.to(self.cfg.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                target_true.append(target.detach().cpu().numpy())
                target_pred.append(output.detach().cpu().numpy())

                if self.clip_grad or self.use_dp:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.clip_value,
                        norm_type=self.clip_norm,
                    )
            
            train_loss /= len(self.dataloader)
            target_true, target_pred = np.concatenate(target_true), np.concatenate(target_pred)
            train_accuracy = float(self.metric(target_true,target_pred))
            
            ## Validation
            if self.cfg.validation == True and self.test_dataloader != None:
                test_loss, test_accuracy = super(PersonalizedClientOptim, self).client_validation()
            else:
                test_loss, test_accuracy = 0, 0
            per_iter_time = time.time() - start_time
            super(PersonalizedClientOptim, self).client_log_content(t+1, per_iter_time, train_loss, train_accuracy, test_loss, test_accuracy)
 
        self.round += 1

        ## Move the model parameter to CPU (if not) for communication
        self.primal_state = copy.deepcopy(self.model.state_dict())
        if (self.cfg.device == "cuda"):            
            for k in self.primal_state:
                self.primal_state[k] = self.primal_state[k].cpu()

        ## Differential Privacy
        if self.use_dp:
            sensitivity = 2.0 * self.clip_value * self.optim_args.lr
            scale_value = sensitivity / self.epsilon
            super(PersonalizedClientOptim, self).laplace_mechanism_output_perturb(scale_value)
            
        ## Save each client model periodically  
        if self.cfg.personalization == True and self.cfg.save_model_state_dict == True and ((self.round) % self.cfg.checkpoints_interval == 0 or self.round== self.cfg.num_epochs):
            save_partial_model_iteration(self.round, self.model, self.cfg, client_id=self.id)

        return self.primal_state
 
