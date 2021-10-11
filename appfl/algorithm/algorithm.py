import copy

"""
This implements a base class for server.
"""
class BaseServer:
    def __init__(self, model, num_clients, device):
        self.model = model
        self.num_clients = num_clients
        self.device = device

    # update global model
    def update(self):
        raise NotImplementedError

    def get_model(self):
        return copy.deepcopy(self.model)


"""
This implements a base class for client.
"""
class BaseClient:
    def __init__(self, model, optimizer, optimizer_args, local_epoch, dataset, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device

    # update local model
    def update(self):
        raise NotImplementedError

    def get_model(self):
        return self.model.state_dict()
