import copy

"""This implements a base class for server."""
 

class BaseServer:
    def __init__(self, weights, model, num_clients, device):
        self.model = model
        self.num_clients = num_clients
        self.device = device
        self.weights = weights

    # update global model
    def update(self):
        raise NotImplementedError

    def get_model(self):
        return copy.deepcopy(self.model)

class BaseClient:
    def __init__(self, id, weight, model, dataloader, device):
        self.id = id
        self.weight = weight
        self.model = model        
        self.dataloader = dataloader
        self.device = device

    # update local model
    def update(self):
        raise NotImplementedError

    def get_model(self):
        return self.model.state_dict()

