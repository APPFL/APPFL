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


class ClientOptimPCA(BaseClient):
    def __init__(
        self, id, weight, model, dataloader, cfg, outfile, test_dataloader, **kwargs
    ):
        super(ClientOptimPCA, self).__init__(
            id, weight, model, dataloader, cfg, outfile, test_dataloader
        )
        self.__dict__.update(kwargs)

        self.loss_fn = eval(self.loss_type)

        self.round = 0

        super(ClientOptimPCA, self).client_log_title()

    def update(self):

        """Inputs for the local model update"""

        self.model.to(self.cfg.device)

        optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)

        """ Multiple local update """
        for t in range(self.num_local_epochs):

            if self.cfg.validation == True and self.test_dataloader != None:
                train_loss, train_accuracy = super(ClientOptim, self).client_validation(
                    self.dataloader
                )
                test_loss, test_accuracy = super(ClientOptim, self).client_validation(
                    self.test_dataloader
                )
                super(ClientOptim, self).client_log_content(
                    t, train_loss, train_accuracy, test_loss, test_accuracy
                )
                ## return to train mode
                self.model.train()

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

        return self.local_state

    def write_result(self, output_filename, W, P, explained_variance_ratio_):

        dir = self.cfg.output_dirname
        if os.path.isdir(dir) == False:
            os.mkdir(dir)

        file_ext = ".txt"
        filename = dir + "/%s%s" % (output_filename, file_ext)
        uniq = 1
        while os.path.exists(filename):
            filename = dir + "/%s_%d%s" % (output_filename, uniq, file_ext)
            uniq += 1

        outfile = open(filename, "w")

        outfile.write("W: (%s, %s) \n" % (W.shape[0], W.shape[1]))
        outfile.write("Ratio: %s \n" % (explained_variance_ratio_))
        outfile.write("Sum: %s \n" % (sum(explained_variance_ratio_)))
        outfile.write("P: (%s, %s) \n" % (P.shape[0], P.shape[1]))

        title = "%10s %10s %10s %10s %10s %10s \n" % (
            "Round",
            "LocalEpoch",
            "TrainLoss",
            "TrainAccu",
            "TestLoss",
            "TestAccu",
        )
        outfile.write(title)

        return outfile, dir
