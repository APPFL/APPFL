import logging

log = logging.getLogger(__name__)

from collections import OrderedDict
from .algorithm import BaseServer, BaseClient

import torch
from torch.optim import *
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import copy


class FedAvgServer(BaseServer):
    def __init__(self, weights, model, num_clients, device, **kwargs):
        super(FedAvgServer, self).__init__(weights, model, num_clients, device)
        self.__dict__.update(kwargs)

    def update(self, local_states: OrderedDict):

        """ Inputs for the global model update """
        global_state = OrderedDict()
        super(FedAvgServer, self).primal_recover_from_local_states(local_states)


        """ global_state calculation """
        for name, _ in self.model.named_parameters():
            tmp = 0.0
            for i in range(self.num_clients):
                ## change device
                self.primal_states[i][name] = self.primal_states[i][name].to(self.device)
                ## computation
                tmp += self.weights[i] * self.primal_states[i][name]

            global_state[name] = tmp


        """ model update """
        self.model.load_state_dict(global_state)



class FedAvgClient(BaseClient):
    def __init__(self, id, weight, model, dataloader, device, **kwargs):
        super(FedAvgClient, self).__init__(id, weight, model, dataloader, device)
        self.__dict__.update(kwargs)
        self.loss_fn = CrossEntropyLoss()

    def update(self):

        self.model.train()
        self.model.to(self.device)

        optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)

        """ Inputs for the local model update """
        ## "global_state" from a server is already stored in 'self.model'

        """ Multiple local update """
        for i in range(self.num_local_epochs):
            for data, target in self.dataloader:

                data = data.to(self.device)
                target = target.to(self.device)

                optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()

                optimizer.step()

        self.primal_state = copy.deepcopy(self.model.state_dict())

        """ Differential Privacy  """
        if self.privacy == True:
            super(FedAvgClient, self).laplace_mechanism_output_perturb()

        """ Update local_state """
        self.local_state = OrderedDict()
        self.local_state["primal"] = copy.deepcopy(self.primal_state)
        self.local_state["dual"] = OrderedDict()
        self.local_state["penalty"] = OrderedDict()
        self.local_state["penalty"][self.id] = 0.0

        return self.local_state
