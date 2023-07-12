import logging
import os
from collections import OrderedDict
from .algorithm import BaseClient

import torch
from torch.optim import *

from torch.utils.data import DataLoader
import copy

import numpy as np
import time
from appfl.compressor import *
from appfl.misc import *


class ClientOptim(BaseClient):
    def __init__(
        self,
        id,
        weight,
        model,
        loss_fn,
        dataloader,
        cfg,
        outfile,
        test_dataloader,
        **kwargs
    ):
        super(ClientOptim, self).__init__(
            id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader
        )
        self.__dict__.update(kwargs)

        self.round = 0
        self.compressor = Compressor(self.cfg)
        self.id = id

        super(ClientOptim, self).client_log_title()

    def update(self):
        """Inputs for the local model update"""

        self.model.to(self.cfg.device)

        optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)

        """ Multiple local update """
        start_time = time.time()
        ## initial evaluation
        if self.cfg.validation == True and self.test_dataloader != None:
            test_loss, test_accuracy = super(ClientOptim, self).client_validation(
                self.test_dataloader
            )
            per_iter_time = time.time() - start_time
            super(ClientOptim, self).client_log_content(
                0, per_iter_time, 0, 0, test_loss, test_accuracy
            )
            ## return to train mode
            self.model.train()

        ## local training
        for t in range(self.num_local_epochs):
            start_time = time.time()
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
            ## Validation
            train_loss = train_loss / len(self.dataloader)
            train_accuracy = 100.0 * train_correct / tmptotal
            if self.cfg.validation == True and self.test_dataloader != None:
                test_loss, test_accuracy = super(ClientOptim, self).client_validation(
                    self.test_dataloader
                )
                per_iter_time = time.time() - start_time
                super(ClientOptim, self).client_log_content(
                    t + 1,
                    per_iter_time,
                    train_loss,
                    train_accuracy,
                    test_loss,
                    test_accuracy,
                )
                ## return to train mode
                self.model.train()

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

        self.primal_state = copy.deepcopy(self.model.state_dict())
        if self.cfg.device == "cuda":
            for k in self.primal_state:
                self.primal_state[k] = self.primal_state[k].cpu()
        start_time = time.time()
        compression_ratio = 0
        flat_params = utils.flatten_model_params(self.model)
        if self.cfg.compressed_weights_client == True:
            compressed_weights_client_arr = self.compressor.compress(
                ori_data=flat_params
            )
            self.cfg.flat_model_size = flat_params.shape
            compression_ratio = (len(flat_params)) / (
                len(compressed_weights_client_arr) * 4
            )
        compress_time = time.time() - start_time
        stats_file = "stats_" + str(self.id) + ".csv"
        with open(stats_file, "a") as f:
            f.write(
                str(self.cfg.dataset)
                + ","
                + str(self.cfg.model)
                + ","
                + str(self.id)
                + ","
                + str(self.cfg.compressed_weights_client)
                + ","
                + str(self.cfg.compressed_weights_server)
                + ","
                + str(compression_ratio)
                + ","
                + self.cfg.compressor
                + ","
                + self.cfg.compressor_error_mode
                + ","
                + str(self.cfg.compressor_error_bound)
                + ","
                + str(flat_params.shape[0])
                + ","
                + str(compress_time)
                + ","
            )

        """ Differential Privacy  """
        if self.epsilon != False:
            sensitivity = 0
            if self.clip_value != False:
                sensitivity = 2.0 * self.clip_value * self.optim_args.lr
            scale_value = sensitivity / self.epsilon
            super(ClientOptim, self).laplace_mechanism_output_perturb(scale_value)

        """ Update local_state """
        self.local_state = OrderedDict()
        self.local_state["primal"] = self.primal_state
        if self.cfg.compressed_weights_client == True:
            self.local_state["primal"] = compressed_weights_client_arr
        self.local_state["dual"] = OrderedDict()
        self.local_state["penalty"] = OrderedDict()
        self.local_state["penalty"][self.id] = 0.0

        return self.local_state
