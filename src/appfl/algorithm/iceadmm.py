import logging

log = logging.getLogger(__name__)

from collections import OrderedDict
from .algorithm import BaseServer, BaseClient

import torch
from torch.optim import *
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import copy


class ICEADMMServer(BaseServer):
    def __init__(self, weights, model, loss_fn, num_clients, device, **kwargs):
        super(ICEADMMServer, self).__init__(weights, model, loss_fn, num_clients, device)
        self.__dict__.update(kwargs)

        self.is_first_iter = 1

    def update(self, local_states: OrderedDict):

        """Inputs for the global model update"""
        self.global_state = copy.deepcopy(self.model.state_dict())
        super(ICEADMMServer, self).primal_recover_from_local_states(local_states)
        super(ICEADMMServer, self).dual_recover_from_local_states(local_states)
        super(ICEADMMServer, self).penalty_recover_from_local_states(local_states)

        """ residual calculation """
        super(ICEADMMServer, self).primal_residual_at_server()
        super(ICEADMMServer, self).dual_residual_at_server()

        total_penalty = 0
        for i in range(self.num_clients):
            total_penalty += self.penalty[i]

        """ global_state calculation """
        for name, param in self.model.named_parameters():
            tmp = 0.0
            for i in range(self.num_clients):
                ## change device
                self.primal_states[i][name] = self.primal_states[i][name].to(
                    self.device
                )
                self.dual_states[i][name] = self.dual_states[i][name].to(self.device)
                ## computation
                tmp += (self.penalty[i] / total_penalty) * self.primal_states[i][
                    name
                ] + (1.0 / total_penalty) * self.dual_states[i][name]

            self.global_state[name] = tmp

        """ model update """
        self.model.load_state_dict(self.global_state)

    def logging_iteration(self, cfg, logger, t):
        if t == 0:
            title = super(ICEADMMServer, self).log_title()
            title = title + "%12s %12s %12s %12s" % (
                "PrimRes",
                "DualRes",
                "RhoMin",
                "RhoMax",
            )
            logger.info(title)

        contents = super(ICEADMMServer, self).log_contents(cfg, t)
        contents = contents + "%12.4e %12.4e %12.4e %12.4e" % (
            self.prim_res,
            self.dual_res,
            min(self.penalty.values()),
            max(self.penalty.values()),
        )
        logger.info(contents)

    def logging_summary(self, cfg, logger):
        super(ICEADMMServer, self).log_summary(cfg, logger)


class ICEADMMClient(BaseClient):
    def __init__(self, id, weight, model, dataloader, device, **kwargs):
        super(ICEADMMClient, self).__init__(id, weight, model, dataloader, device)
        self.__dict__.update(kwargs)
        self.loss_fn = eval(self.loss_type)

        """ 
        At initial, (1) primal_state = global_state, (2) dual_state = 0
        """
        self.model.to(device)
        for name, param in model.named_parameters():
            self.primal_state[name] = param.data
            self.dual_state[name] = torch.zeros_like(param.data)

        self.penalty = kwargs["init_penalty"]
        self.proximity = kwargs["init_proximity"]
        self.is_first_iter = 1

    def update(self):

        self.model.train()
        self.model.to(self.device)

        optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)

        """ Inputs for the local model update """
        global_state = copy.deepcopy(self.model.state_dict())

        """ Adaptive Penalty (Residual Balancing) """
        if self.residual_balancing.res_on == True:
            prim_res = super(ICEADMMClient, self).primal_residual_at_client(
                global_state
            )
            dual_res = super(ICEADMMClient, self).dual_residual_at_client()
            super(ICEADMMClient, self).residual_balancing(prim_res, dual_res)

        """ Multiple local update """
        for i in range(self.num_local_epochs):
            for data, target in self.dataloader:

                self.model.load_state_dict(self.primal_state)

                if (
                    self.residual_balancing.res_on == True
                    and self.residual_balancing.res_on_every_update == True
                ):
                    prim_res = super(ICEADMMClient, self).primal_residual_at_client(
                        global_state
                    )
                    dual_res = super(ICEADMMClient, self).dual_residual_at_client()
                    super(ICEADMMClient, self).residual_balancing(prim_res, dual_res)

                data = data.to(self.device)
                target = target.to(self.device)

                if self.accum_grad == False:
                    optimizer.zero_grad()

                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()

                if self.clip_value != False:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.clip_value,
                        norm_type=self.clip_norm,
                    )

                ## STEP: Update primal and dual
                coefficient = 1
                if self.coeff_grad == True:
                    coefficient = (
                        self.weight * len(target) / len(self.dataloader.dataset)
                    )

                self.iceadmm_step(coefficient, global_state)

        """ Differential Privacy  """
        if self.epsilon != False:
            sensitivity = 0
            if self.clip_value != False:
                sensitivity = 2.0 * self.clip_value / self.penalty
            scale_value = sensitivity / self.epsilon
            super(ICEADMMClient, self).laplace_mechanism_output_perturb(scale_value)

        """ Update local_state """
        self.local_state = OrderedDict()
        self.local_state["primal"] = copy.deepcopy(self.primal_state)
        self.local_state["dual"] = copy.deepcopy(self.dual_state)
        self.local_state["penalty"] = OrderedDict()
        self.local_state["penalty"][self.id] = self.penalty

        return self.local_state

    def iceadmm_step(self, coefficient, global_state):
        for name, param in self.model.named_parameters():

            grad = param.grad * coefficient
            ## Update primal
            self.primal_state[name] = self.primal_state[name] - (
                self.penalty * (self.primal_state[name] - global_state[name])
                + grad
                + self.dual_state[name]
            ) / (self.weight * self.proximity + self.penalty)
            ## Update dual
            self.dual_state[name] = self.dual_state[name] + self.penalty * (
                self.primal_state[name] - global_state[name]
            )
