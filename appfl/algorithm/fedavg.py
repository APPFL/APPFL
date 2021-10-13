import logging

log = logging.getLogger(__name__)

from collections import OrderedDict
from .algorithm import BaseServer, BaseClient

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader


class FedAvgServer(BaseServer):
    def __init__(self, model, num_clients, device, dataloader=None):
        super(FedAvgServer, self).__init__(model, num_clients, device)

        self.dataloader = dataloader
        if self.dataloader is not None:
            self.loss_fn = CrossEntropyLoss()
        else:
            self.loss_fn = None

    # update global model
    def update(self, local_states: OrderedDict):
        update_state = OrderedDict()

        for k, state in local_states.items():
            for key in self.model.state_dict().keys():
                if k == 0:
                    update_state[key] = state[key] / self.num_clients
                else:
                    update_state[key] += state[key] / self.num_clients

        self.model.load_state_dict(update_state)

    # NOTE: this is only for testing purpose.
    def validation(self):
        if self.loss_fn is None or self.dataloader is None:
            return 0.0, 0.0

        self.model.to(self.device)
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for img, target in self.dataloader:
                img = img.to(self.device)
                target = target.to(self.device)
                logits = self.model(img)
                test_loss += self.loss_fn(logits, target).item()
                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        # FIXME: do we need to sent the model to cpu again?
        # self.model.to("cpu")
        test_loss = test_loss / len(self.dataloader)
        accuracy = 100.0 * correct / len(self.dataloader.dataset)

        return test_loss, accuracy


class FedAvgClient(BaseClient):
    def __init__(
        self, id, model, optimizer, optimizer_args, dataloader, device, **kwargs
    ):
        super(FedAvgClient, self).__init__(
            id, model, optimizer, optimizer_args, dataloader, device
        )
        self.loss_fn = CrossEntropyLoss()
        self.__dict__.update(kwargs)

    # update local model
    def update(self):
        self.model.train()
        self.model.to(self.device)
        optimizer = self.optimizer(self.model.parameters(), **self.optimizer_args)

        for i in range(self.num_local_epochs):
            log.info(f"[Client ID: {self.id+1: 03}, Local epoch: {i+1: 04}]")
            for data, target in self.dataloader:
                data = data.to(self.device)
                target = target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()
                optimizer.step()
