import os
import copy
import time
import torch
from torch.optim import *
from .algorithm import BaseClient
from collections import OrderedDict

class ClientStepOptim(BaseClient):
    """This client optimizer which perform updates for certain number of steps/batches in each training round."""
    def __init__(
        self, id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader, metric, **kwargs
    ):
        super(ClientStepOptim, self).__init__(id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader)
        self.__dict__.update(kwargs)
        self.metric = metric
        super(ClientStepOptim, self).client_log_title()

    def update(self):
        self.model.to(self.cfg.device)
        optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)
        start_time=time.time()
        ## Initial evaluation
        if self.cfg.validation == True and self.test_dataloader != None:
            test_loss, test_accuracy = super(ClientStepOptim, self).client_validation(self.test_dataloader, self.metric)
            per_iter_time = time.time() - start_time
            super(ClientStepOptim, self).client_log_content(0, per_iter_time, 0, 0, test_loss, test_accuracy)
            ## Return to train mode
            self.model.train()        

        ## Local training
        data_iter = iter(self.dataloader)
        start_time = time.time()
        epoch = 1
        for _ in range(self.num_local_steps):
            try:
                data, target = next(data_iter)
            except: # End of one local epoch
                ## Validation
                if self.cfg.validation == True and self.test_dataloader != None:
                    train_loss, train_accuracy = super(ClientStepOptim, self).client_validation(self.dataloader, self.metric)
                    test_loss, test_accuracy = super(ClientStepOptim, self).client_validation(self.test_dataloader, self.metric)
                    per_iter_time = time.time() - start_time
                    super(ClientStepOptim, self).client_log_content(epoch, per_iter_time, train_loss, train_accuracy, test_loss, test_accuracy)
                    self.model.train()
                start_time = time.time()
                epoch += 1

                ## save model.state_dict()
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
            optimizer.step()
            if self.clip_value != False:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value, norm_type=self.clip_norm)
        self.round += 1
        ## Move the model to CPU
        self.primal_state = copy.deepcopy(self.model.state_dict())
        if (self.cfg.device == "cuda"):            
            for k in self.primal_state:
                self.primal_state[k] = self.primal_state[k].cpu()

        ## Differential Privacy
        if self.epsilon != False:
            sensitivity = 0
            if self.clip_value != False:
                sensitivity = 2.0 * self.clip_value * self.optim_args.lr
            scale_value = sensitivity / self.epsilon
            super(ClientStepOptim, self).laplace_mechanism_output_perturb(scale_value)

        ## Update local_state
        self.local_state = OrderedDict()
        self.local_state["primal"] = self.primal_state
        self.local_state["dual"] = OrderedDict()
        self.local_state["penalty"] = OrderedDict()
        self.local_state["penalty"][self.id] = 0.0

        return self.local_state
 
