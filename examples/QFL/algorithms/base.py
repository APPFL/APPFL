from abc import ABC, abstractmethod

class BaseFLAlgorithm(ABC):
    def __init__(self, cfg, server, clients, client_ids, override_steps=None):
        self.cfg = cfg
        self.server = server
        self.clients = clients
        self.client_ids = client_ids
        self.override_steps = override_steps

    @abstractmethod
    def run(self):
        ...
