import copy

from collections import OrderedDict
import torch


class BaseServer:
    """Abstract class of PPFL algorithm for server that aggregates and updates model parameters.

    Args:
        weight: aggregation weight assigned to each client
        model: (nn.Module): torch neural network model to train
        num_clients (int): the number of clients
        device (str): device for computation
    """
    def __init__(self, weights, model, num_clients, device):
        self.model = model
        self.num_clients = num_clients
        self.device = device
        self.weights = weights    
        self.penalty = OrderedDict()
        self.primal_states = OrderedDict()
        self.dual_states = OrderedDict()
        for i in range(num_clients):
            self.primal_states[i] = OrderedDict()
            self.dual_states[i] = OrderedDict()

    def update(self):
        """ Update global model parameters """
        raise NotImplementedError

    def get_model(self):
        """ Get the model

        Return:
            nn.Module: a deepcopy of self.model
        """
        return copy.deepcopy(self.model)
    
    def primal_recover_from_local_states(self, local_states):
        for _, states in enumerate(local_states):            
            if states is not None:
                for sid, state in states.items():
                    self.primal_states[sid] = copy.deepcopy(state["primal"])

    def dual_recover_from_local_states(self, local_states):
        for _, states in enumerate(local_states):
            if states is not None:
                for sid, state in states.items():
                    self.dual_states[sid] = copy.deepcopy(state["dual"])

    def penalty_recover_from_local_states(self, local_states):
        for _, states in enumerate(local_states):
            if states is not None:
                for sid, state in states.items():                    
                    self.penalty[sid] = copy.deepcopy(state["penalty"][sid])        


"""This implements a base class for clients."""


class BaseClient:
    """Abstract class of PPFL algorithm for client that trains local model.

    Args:
        id: unique ID for each client
        weight: aggregation weight assigned to each client
        model: (nn.Module): torch neural network model to train
        dataloader: PyTorch data loader
        device (str): device for computation
    """
    def __init__(self, id, weight, model, dataloader, device):
        self.id = id
        self.weight = weight
        self.model = model
        self.dataloader = dataloader
        self.device = device

        self.primal_state = OrderedDict()
        self.dual_state = OrderedDict()

    def update(self):
        """ Update local model parameters """
        raise NotImplementedError

    def get_model(self):
        """ Get the model

        Return:
            the ``state_dict`` of local model
        """
        return self.model.state_dict()


    """ 
    Differential Privacy 
        (Laplacian mechanism) 
        - Noises from a Laplace dist. with zero mean and "scale_value" are added to the primal_state 
        - Variance = 2*(scale_value)^2
        - scale_value = sensitivty/epsilon, where sensitivity is determined by data, algorithm.
    """        

    def laplace_mechanism_output_perturb(self,scale_value):    
        """Differential privacy for output perturbation based on Laplacian distribution. 
        This output perturbation adds Laplace noise to ``primal_state``.

        Args:
            scale_value: scaling vector to control the variance of Laplacian distribution
        """   

        for name, param in self.model.named_parameters():
            mean = torch.zeros_like(param.data)
            scale = torch.zeros_like(param.data) + scale_value
            m = torch.distributions.laplace.Laplace(mean, scale)
            self.primal_state[name] += m.sample()
