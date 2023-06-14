import logging
from collections import OrderedDict

from .server_federated import FedServer
from .algorithm import BaseClient


class FDAServer(FedServer):
    def __init__(self, weights, model, num_clients, device, **kwargs):
        super(FDAServer, self).__init__(weights, model, num_clients, device)
        self.__dict__.update(kwargs)
        # Any additional initialization
        pass

    def update(self, local_states: OrderedDict):
        # Implement new server update function
        pass

class FDAClient(BaseClient):
    def __init__(self, id, weight, model, dataloader, device, **kwargs):
        super(FDAClient, self).__init__(id, weight, model, dataloader, device)
        self.__dict__.update(kwargs)
        # Any additional initialization
        pass

    def update(self):
        # Implement new client update function
        pass