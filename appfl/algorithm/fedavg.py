from collections import OrderedDict
from .algorithm import BaseServer, BaseClient

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

class FedAvgServer(BaseServer):
    def __init__(self, model, num_clients, device):

        self.model = model
        self.num_clients = num_clients
        self.device = device

    # update global model
    def update(self, local_states):
        print("[Server] update")
        update_state = OrderedDict()

        for k, state in enumerate(local_states):
            for key in self.model.state_dict().keys():
                if k == 0:
                    update_state[key] = state[key] / self.num_clients
                else:
                    update_state[key] += state[key] / self.num_clients

        self.model.load_state_dict(update_state)

    # TODO: this is only for testing purpose.
    def validation(self):
        print("[Server] validation")
        test_loss = 0.0
        accuracy = 0.0
        return test_loss, accuracy


class FedAvgClient(BaseClient):
    def __init__(self, model, optimizer, optimizer_args, local_epoch, dataset, device):
        self.model = model
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args
        self.local_epoch = local_epoch
        self.dataloader = DataLoader(dataset, num_workers=0, batch_size=4, shuffle=True)
        self.device = device

        self.loss_fn = CrossEntropyLoss()

    # update local model
    def update(self):
        print("[Client] update")
        self.model.train()
        self.model.to(self.device)
        optimizer = self.optimizer(self.model.parameters(), **self.optimizer_args)

        for i in range(self.local_epoch):
            for data, target in self.dataloader:
                data = data.to(self.device)
                target = target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()
                optimizer.step()
