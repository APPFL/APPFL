import os
import copy
import time
import torch
from torch.optim import *
from .fl_base import BaseClient

class GlobusComputeClientOptim(BaseClient):
    """GlobusComputeClientOptim is the ClientOptim accompanied with a ClientLogger for recording the training process."""
    def __init__(self, id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader, metric, **kwargs):
        super(GlobusComputeClientOptim, self).__init__(id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader, metric)
        super(GlobusComputeClientOptim, self).client_log_title()
        self.__dict__.update(kwargs)

    def update(self, cli_logger):
        self.model.to(self.cfg.device)
        optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)
        ## initial evaluation
        if self.cfg.validation == True and self.test_dataloader != None:
            start_time=time.time()
            cli_logger.start_timer("val_before_update_val_set")
            test_loss, test_accuracy =self.client_validation()
            cli_logger.add_info(
                "val_before_update_val_set",{
                    "val_loss": test_loss, "val_acc": test_accuracy
                }
            )
            cli_logger.stop_timer("val_before_update_val_set")
            per_iter_time = time.time() - start_time
            super(GlobusComputeClientOptim, self).client_log_content(
                0, per_iter_time, 0, 0, test_loss, test_accuracy
            )     

        ## local training 
        for t in range(self.num_local_epochs):
            cli_logger.start_timer("train_one_epoch", t)
            for data, target in self.dataloader:             
                data = data.to(self.cfg.device)
                target = target.to(self.cfg.device)
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
            cli_logger.stop_timer("train_one_epoch", t)

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
        self.primal_state = copy.deepcopy(self.model.to('cpu').state_dict())

        ## Differential Privacy 
        if self.epsilon != False:
            sensitivity = 0
            if self.clip_value != False:
                sensitivity = 2.0 * self.clip_value * self.optim_args.lr
            scale_value = sensitivity / self.epsilon
            super(GlobusComputeClientOptim, self).laplace_mechanism_output_perturb(scale_value)
        
        if cli_logger is not None:
            return self.primal_state, cli_logger
        else:
            return self.primal_state
 
