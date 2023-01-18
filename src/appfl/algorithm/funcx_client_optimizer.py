import logging
import os
from collections import OrderedDict

from appfl.misc.logging import ClientLogger
from .algorithm import BaseClient

import torch
from torch.optim import *
from torch.nn import functional as F

from torch.utils.data import DataLoader
import copy

import numpy as np
import time

from sklearn import metrics
class FuncxClientOptim(BaseClient):
    def __init__(
        self, id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader, **kwargs
    ):
        super(FuncxClientOptim, self).__init__(
            id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader
        )
        self.__dict__.update(kwargs)

        self.round = 0

        super(FuncxClientOptim, self).client_log_title()

    def client_validation(self, dataloader):

        if self.loss_fn is None or dataloader is None:
            return 0.0, 0.0

        self.model.to(self.cfg.device)
        self.model.eval()
        loss = 0
        correct = 0
        tmpcnt = 0
        tmptotal = 0

        preds   = []
        targets = []
        with torch.no_grad():
            for img, target in dataloader:
                tmpcnt += 1
                tmptotal += len(target)
                img     = img.to(self.cfg.device)
                target  = target.to(self.cfg.device)
                output  = self.model(img)
                loss    += self.loss_fn(output, target).item()
                
                if output.shape[1] == 1:
                    pred = torch.round(output)
                else:
                    pred = F.softmax(output)
                    # pred = output.argmax(dim=1, keepdim=True)
                preds.append(pred.cpu().detach().numpy())
                targets.append(target.cpu().detach().numpy())

        # FIXME: do we need to sent the model to cpu again?
        # self.model.to("cpu")
        
        targets = np.concatenate(targets)
        preds   = np.concatenate(preds)

        preds_binary = preds.argmax(axis=1) 
        acc = (preds_binary == targets).mean()
        prec,rec, f1, sprt  = metrics.precision_recall_fscore_support(preds_binary, targets, average="binary")
        auc = metrics.roc_auc_score(targets, preds[:, 1])
        # correct += pred.eq(preds.argmax(dim=1, keepdim=True).view_as(pred)).sum().item()    
        return loss, {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'auc': auc}

    def update(self, cli_logger):
        
        """Inputs for the local model update"""
        self.model.to(self.cfg.device)

        optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)
 

        """ Multiple local update """
        start_time=time.time()
        ## initial evaluation
        if self.cfg.validation == True and self.test_dataloader != None:
            cli_logger.start_timer("val_before_update_val_set")
            test_loss, test_accuracy = super(FuncxClientOptim, self).client_validation(
                self.test_dataloader
            )
            cli_logger.add_info(
                    "val_before_update_val_set",{
                        "val_loss": test_loss, "val_acc": test_accuracy
                    }
                )
            cli_logger.stop_timer("val_before_update_val_set")
            per_iter_time = time.time() - start_time
            super(FuncxClientOptim, self).client_log_content(
                0, per_iter_time, 0, 0, test_loss, test_accuracy
            )
            ## return to train mode
            self.model.train()        

        ## local training 
        for t in range(self.num_local_epochs):
            cli_logger.start_timer("train_one_epoch", t)
            train_loss = 0
            train_correct = 0            
            tmptotal = 0
            for data, target in self.dataloader:             
                tmptotal += len(target)
                data = data.to(self.cfg.device)
                target = target.to(self.cfg.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                if output.shape[1] == 1:
                    pred = torch.round(output)
                else:
                    pred = output.argmax(dim=1, keepdim=True)
                train_correct += pred.eq(target.view_as(pred)).sum().item()

                if self.clip_value != False:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.clip_value,
                        norm_type=self.clip_norm,
                    )
            cli_logger.stop_timer("train_one_epoch", t)

            ## Validation
            # train_loss = train_loss / len(self.dataloader)
            # train_accuracy = 100.0 * train_correct / tmptotal
            # if self.cfg.validation == True and self.test_dataloader != None:
            #     test_loss, test_accuracy = super(FuncxClientOptim, self).client_validation(
            #         self.test_dataloader
            #     )
            #     per_iter_time = time.time() - start_time
            #     super(FuncxClientOptim, self).client_log_content(
            #         t+1, per_iter_time, train_loss, train_accuracy, test_loss, test_accuracy
            #     )
            #     ## return to train mode
            #     self.model.train()

            ## save model.state_dict()
            if self.cfg.save_model_state_dict == True:
                path = self.cfg.output_dirname + "/client_%s" % (self.id)
                if not os.path.exists(path):
                    os.makedirs(path)
                torch.save(
                    self.model.state_dict(),
                    os.path.join(path, "%s_%s.pt" % (self.round, t)),
                )

            # if (t == self.num_local_epochs-1)  and self.test_dataloader != None:
            #     cli_logger.start_timer("val_after_update_val_set", t)
            #     test_loss, test_accuracy = super(
            #         FuncxClientOptim, self
            #     ).client_validation(self.test_dataloader)
            #     print(test_loss, test_accuracy)
            #     cli_logger.stop_timer("val_after_update_val_set", t)
                
            #     cli_logger.add_info(
            #             "val_after_update_val_set",{
            #                 "val_loss": test_loss, "val_accuracy": test_accuracy
            #             }
            #         )
            #     self.model.train()

        self.round += 1

        self.primal_state = copy.deepcopy(self.model.state_dict())

        """ Differential Privacy  """
        if self.epsilon != False:
            sensitivity = 0
            if self.clip_value != False:
                sensitivity = 2.0 * self.clip_value * self.optim_args.lr
            scale_value = sensitivity / self.epsilon
            super(FuncxClientOptim, self).laplace_mechanism_output_perturb(scale_value)

        """ Update local_state """
        self.local_state = OrderedDict()
        self.local_state["primal"] = copy.deepcopy(self.primal_state)
        self.local_state["dual"] = OrderedDict()
        self.local_state["penalty"] = OrderedDict()
        self.local_state["penalty"][self.id] = 0.0
        
        if cli_logger is not None:
            return self.local_state, cli_logger
        else:
            return self.local_state
 
