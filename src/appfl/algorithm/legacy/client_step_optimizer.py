import os
import copy
import time
import torch
import numpy as np
from torch.optim import *
from .fl_base import BaseClient
from appfl.misc import deprecated

@deprecated("Imports from appfl.algorithm is deprecated and will be removed in the future. Please use appfl.algorithm.trainer instead.")
class ClientStepOptim(BaseClient):
    """This client optimizer which perform updates for certain number of steps/batches in each training round."""
    def __init__(
        self, id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader, metric, **kwargs
    ):
        super(ClientStepOptim, self).__init__(id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader, metric)
        self.__dict__.update(kwargs)
        super(ClientStepOptim, self).client_log_title()

    def update(self):
        self.model.to(self.cfg.device)
        optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)
        
        ## Initial evaluation
        if self.cfg.validation == True and self.test_dataloader != None:
            start_time=time.time()
            test_loss, test_accuracy = super(ClientStepOptim, self).client_validation()
            per_iter_time = time.time() - start_time
            super(ClientStepOptim, self).client_log_content(0, per_iter_time, 0, 0, test_loss, test_accuracy)

        ## Local training
        data_iter = iter(self.dataloader)
        start_time = time.time()
        epoch = 1
        train_loss, target_true, target_pred = 0, [], []
        for _ in range(self.num_local_steps):
            try:
                data, target = next(data_iter)
            except: # End of one local epoch
                train_loss /= len(self.dataloader)
                target_true, target_pred = np.concatenate(target_true), np.concatenate(target_pred)
                train_accuracy = float(self.metric(target_true, target_pred))
                ## Validation
                if self.cfg.validation == True and self.test_dataloader != None:
                    test_loss, test_accuracy = super(ClientStepOptim, self).client_validation()
                else:
                    test_loss, test_accuracy = 0, 0
                per_iter_time = time.time() - start_time
                super(ClientStepOptim, self).client_log_content(epoch, per_iter_time, train_loss, train_accuracy, test_loss, test_accuracy)
                
                epoch += 1

                ## save model state dict
                if self.cfg.save_model_state_dict == True:
                    path = self.cfg.output_dirname + "/client_%s" % (self.id)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(path, "%s_%s.pt" % (self.round, epoch)),
                    )
                ## Reset the data iterator
                data_iter = iter(self.dataloader)
                data, target = next(data_iter)

                ## Reset the train metric
                train_loss, target_true, target_pred = 0, [], []
                start_time = time.time()

            data = data.to(self.cfg.device)
            target = target.to(self.cfg.device)
            optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            target_true.append(target.detach().cpu().numpy())
            target_pred.append(output.detach().cpu().numpy())
            train_loss += loss.item()
            if self.clip_grad or self.use_dp:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value, norm_type=self.clip_norm)
            optimizer.step()
        
        self.round += 1

        ## Differential Privacy
        self.primal_state = copy.deepcopy(self.model.state_dict())
        if self.use_dp:
            sensitivity = 2.0 * self.clip_value * self.optim_args.lr
            scale_value = sensitivity / self.epsilon
            super(ClientStepOptim, self).laplace_mechanism_output_perturb(scale_value)

        ## Move the model parameter to CPU (if not) for communication
        if (self.cfg.device == "cuda"):            
            for k in self.primal_state:
                self.primal_state[k] = self.primal_state[k].cpu()

        return self.primal_state
 
