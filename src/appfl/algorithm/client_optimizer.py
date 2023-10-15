import os
import copy
import time
import torch
import numpy as np
from torch.optim import *
from .fl_base import BaseClient

class ClientOptim(BaseClient):
    """This client optimizer which perform updates for certain number of epochs in each training round."""
    def __init__(
        self, id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader, metric, **kwargs
    ):
        super(ClientOptim, self).__init__(id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader, metric)
        self.__dict__.update(kwargs)
        super(ClientOptim, self).client_log_title()

    def update(self):
        self.model.to(self.cfg.device)
        optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)

        ## Initial evaluation
        if self.cfg.validation == True and self.test_dataloader != None:
            start_time=time.time()
            test_loss, test_accuracy = super(ClientOptim, self).client_validation()
            per_iter_time = time.time() - start_time
            super(ClientOptim, self).client_log_content(0, per_iter_time, 0, 0, test_loss, test_accuracy)    

        ## Local training 
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
                target_true.append(target.detach().cpu().numpy())
                target_pred.append(output.detach().cpu().numpy())
                train_loss += loss.item()
                if self.clip_grad or self.use_dp:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.clip_value,
                        norm_type=self.clip_norm,
                    )

            train_loss /= len(self.dataloader)
            target_true, target_pred = np.concatenate(target_true), np.concatenate(target_pred)
            train_accuracy = float(self.metric(target_true, target_pred))
            
            ## Validation
            if self.cfg.validation == True and self.test_dataloader != None:
                test_loss, test_accuracy = super(ClientOptim, self).client_validation()
            else:
                test_loss, test_accuracy = 0, 0
            per_iter_time = time.time() - start_time
            super(ClientOptim, self).client_log_content(t+1, per_iter_time, train_loss, train_accuracy, 0, 0)

            ## save model.state_dict()
            if self.cfg.save_model_state_dict == True:
                path = self.cfg.output_dirname + "/client_%s" % (self.id)
                if not os.path.exists(path):
                    os.makedirs(path, exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(path, "%s_%s.pt" % (self.round, t)))
 
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
            super(ClientOptim, self).laplace_mechanism_output_perturb(scale_value)

        return self.primal_state
 