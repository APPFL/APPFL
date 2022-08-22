import logging
import os
from collections import OrderedDict

from appfl.misc.logging import ClientLogger
from .algorithm import BaseClient

import torch
from torch.optim import *

from torch.utils.data import DataLoader
import copy

import numpy as np
import time


class ClientOptim(BaseClient):
    def __init__(
        self, id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader, **kwargs
    ):
        super(ClientOptim, self).__init__(
            id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader
        )
        self.__dict__.update(kwargs)

        self.round = 0

        super(ClientOptim, self).client_log_title()

    def update(self, cli_logger):
        
        """Inputs for the local model update"""
        self.model.to(self.cfg.device)

        optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)

        """ Multiple local update """
        start_time=time.time()

        for t in range(self.num_local_epochs):
            
            if (t == 0) and self.test_dataloader != None: # For now, just run the evaluation once, before training
                ## validate on train set
                # cli_logger.start_timer("val_before_update_train_set", t)
                # train_loss, train_accuracy = super(ClientOptim, self).client_validation(
                #     self.dataloader
                # )
                # cli_logger.add_info(
                #     "val_before_update_train_set",{
                #         "train_loss": train_loss, "train_acc": train_accuracy
                #     }
                # )
                # cli_logger.stop_timer("val_before_update_train_set", t)

                ## validate on val set
                cli_logger.start_timer("val_before_update_val_set", t)
                test_loss, test_accuracy = super(ClientOptim, self).client_validation(
                    self.test_dataloader
                )
                cli_logger.add_info(
                    "val_before_update_val_set",{
                        "val_loss": test_loss, "val_acc": test_accuracy
                    }
                )
                cli_logger.stop_timer("val_before_update_val_set", t)
                
                # per_iter_time = time.time() - start_time
                # super(ClientOptim, self).client_log_content(
                #     t, per_iter_time, train_loss, train_accuracy, test_loss, test_accuracy
                # )
                ## return to train mode
                self.model.train()

            # Do training for one epoch
            cli_logger.start_timer("train_one_epoch", t)
            for i, (data, target) in enumerate(self.dataloader):
                self.outfile.write("epoch [%d][%d/%d] %s" % (t, i+1, len(self.dataloader), str(data.shape)))
                self.outfile.flush()
                data = data.to(self.cfg.device)
                target = target.to(self.cfg.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                print(loss)
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

            if (t == self.num_local_epochs-1)  and self.test_dataloader != None:
                # cli_logger.start_timer("val_after_update_train_set", t)
                # train_loss, train_accuracy = super(
                #     ClientOptim, self
                # ).client_validation(self.dataloader)
                # cli_logger.stop_timer("val_after_update_train_set", t)
                
                # Add validation results at client
                # cli_logger.add_info(
                #         "val_after_update_train_set",{
                #             "train_loss": train_loss, "train_accuracy": train_accuracy
                #         }
                #     )

                cli_logger.start_timer("val_after_update_val_set", t)
                test_loss, test_accuracy = super(
                    ClientOptim, self
                ).client_validation(self.test_dataloader)
                cli_logger.stop_timer("val_after_update_val_set", t)
                
                cli_logger.add_info(
                        "val_after_update_val_set",{
                            "val_loss": test_loss, "val_accuracy": test_accuracy
                        }
                    )
                self.model.train()
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
        
        if cli_logger is not None:
            return self.local_state, cli_logger
        else:
            return self.local_state
 
