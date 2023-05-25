import logging
import os
from collections import OrderedDict
from .algorithm import BaseServer, BaseClient

import torch
from torch.optim import *

from torch.utils.data import DataLoader
import copy

import numpy as np
import time


class FedresServer(BaseServer):
    def __init__(self, weights, model, loss_fn, num_clients, device, **kwargs):
        super(FedresServer, self).__init__(weights, model, loss_fn, num_clients, device)
        self.__dict__.update(kwargs)
        self.logger = logging.getLogger(__name__)

    def update(self, local_states: OrderedDict):
        """Inputs for the global model update"""
        self.global_state = copy.deepcopy(self.model.state_dict())
        super(FedresServer, self).primal_recover_from_local_states(local_states)

        """ change device """
        for i in range(self.num_clients):
            for name in self.model.state_dict():
                self.primal_states[i][name] = self.primal_states[i][name].to(
                    self.device
                )

        """ global_state calculation """
        xrt_idx = self.num_clients - 1

        y = OrderedDict()
        for name, param in self.model.named_parameters():
            y[name] = torch.zeros_like(param.data)
            for i in range(self.num_clients):
                if i < xrt_idx:
                    y[name] = y[name] + self.coeff[i] * self.primal_states[i][name]

            # compute the global_state to broadcast
            self.global_state[name] = 0.5 * (
                self.primal_states[xrt_idx][name] - y[name]
            )

        """ model update """
        self.model.load_state_dict(self.global_state)

    def logging_iteration(self, cfg, logger, t):
        if t == 0:
            title = super(FedresServer, self).log_title()
            logger.info(title)

        contents = super(FedresServer, self).log_contents(cfg, t)
        logger.info(contents)

    def logging_summary(self, cfg, logger):
        print("-----DONE-----")


class FedresClient(BaseClient):
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
        super(FedresClient, self).__init__(
            id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader
        )
        self.__dict__.update(kwargs)

        self.round = 0
        self.local_state_prev = OrderedDict()
        for name, param in self.model.named_parameters():
            self.local_state_prev[name] = param.data
        self.global_state = OrderedDict()
        for name, param in self.model.named_parameters():
            self.global_state[name] = param.data

        super(FedresClient, self).client_log_title()

    def update(self):
        """Inputs for the local model update"""
        self.model.to(self.cfg.device)

        if self.round > 0:
            for (
                name,
                param,
            ) in (
                self.model.named_parameters()
            ):  ## model parameters received from the server
                self.global_state[name] = param.data

            if self.id < self.cfg.num_clients - 1:  ## initial points for XRF clients
                for name, param in self.model.named_parameters():
                    param.data = (
                        self.local_state_prev[name]
                        + self.cfg.fed.args.coeff[self.id] * self.global_state[name]
                    )

            else:  ## initial points for XRT client
                for name, param in self.model.named_parameters():
                    param.data = self.local_state_prev[name] - self.global_state[name]

        """ Multiple local update """
        optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        ## local training
        for t in range(self.num_local_epochs):
            start_time = time.time()
            train_loss = 0
            for data, target in self.dataloader:
                data = data.to(self.cfg.device)
                target = target.to(self.cfg.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()

            per_iter_time = time.time() - start_time

            super(FedresClient, self).client_log_content(
                t + 1, per_iter_time, train_loss, 0.0, 0.0, 0.0
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

        self.local_state_prev = copy.deepcopy(self.model.state_dict())

        """ Update local_state """
        self.local_state = OrderedDict()
        self.local_state["primal"] = self.local_state_prev
        self.local_state["dual"] = OrderedDict()
        self.local_state["penalty"] = OrderedDict()
        self.local_state["penalty"][self.id] = 0.0

        return self.local_state
