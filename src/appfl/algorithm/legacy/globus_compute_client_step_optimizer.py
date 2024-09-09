import os
import copy
import time
import torch
from torch.optim import *
from .fl_base import BaseClient
from appfl.misc import compute_gradient, deprecated

@deprecated("Imports from appfl.algorithm is deprecated and will be removed in the future. Please use appfl.algorithm.trainer instead.")
class GlobusComputeClientStepOptim(BaseClient):
    """GlobusComputeClientStepOptim is the ClientStepOptim accompanied with a ClientLogger for recording the training process."""
    def __init__(self, id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader, metric, global_epoch=0, send_gradient=False, **kwargs):
        super(GlobusComputeClientStepOptim, self).__init__(id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader, metric)
        super(GlobusComputeClientStepOptim, self).client_log_title()
        self.round = global_epoch
        self.send_gradient = send_gradient
        self.__dict__.update(kwargs)

    def update(self, cli_logger):
        start_time=time.time()
        if self.send_gradient:
            original_model = copy.deepcopy(self.model.state_dict())
        self.model.to(self.cfg.device)
        optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)  

        ## local training 
        epoch = 0
        data_iter = iter(self.dataloader)
        cli_logger.start_timer("Train")
        for _ in range(self.num_local_steps):
            try:
                data, target = next(data_iter)
            except:
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
            data = data.to(self.cfg.device)
            target = target.to(self.cfg.device)
            optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            if self.clip_grad or self.use_dp:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value, norm_type=self.clip_norm)
            optimizer.step()
        cli_logger.stop_timer("Train")

        ## client evaluation
        if self.cfg.validation == True and self.test_dataloader != None:
            cli_logger.start_timer("Validation")
            test_loss, test_accuracy =self.client_validation()
            cli_logger.add_info(
                "Validation",{
                    "val_loss": test_loss, "val_acc": test_accuracy
                }
            )
            cli_logger.stop_timer("Validation")
            per_iter_time = time.time() - start_time
            super(GlobusComputeClientStepOptim, self).client_log_content(0, per_iter_time, 0, 0, test_loss, test_accuracy)   

        ## Differential Privacy 
        self.primal_state = copy.deepcopy(self.model.state_dict())
        if self.use_dp:
            sensitivity = 2.0 * self.clip_value * self.optim_args.lr
            scale_value = sensitivity / self.epsilon
            super(GlobusComputeClientStepOptim, self).laplace_mechanism_output_perturb(scale_value)
        
        ## Move the model parameter to CPU (if not) for communication
        if (self.cfg.device == "cuda"):            
            for k in self.primal_state:
                self.primal_state[k] = self.primal_state[k].cpu()
        
        if self.send_gradient:
            return compute_gradient(original_model, self.model), cli_logger
        else:
            return self.primal_state, cli_logger
 
