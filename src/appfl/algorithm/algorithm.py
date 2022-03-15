import copy
from typing import Dict, Tuple

from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class BaseServer:
    """Abstract class of PPFL algorithm for server that aggregates and updates model parameters.

    Args:
        weight (Dict): aggregation weight assigned to each client
        model (nn.Module): torch neural network model to train
        num_clients (int): the number of clients
        device (str): device for computation
    """

    def __init__(self, weights: OrderedDict, model: nn.Module, num_clients: int, device):        
        self.model = model
        self.num_clients = num_clients
        self.device = device
        self.weights = copy.deepcopy(weights)
        self.penalty = OrderedDict()
        self.primal_states = OrderedDict()
        self.dual_states = OrderedDict()
        self.primal_states_curr = OrderedDict()
        self.primal_states_prev = OrderedDict()
        for i in range(num_clients):
            self.primal_states[i] = OrderedDict()
            self.dual_states[i] = OrderedDict()
            self.primal_states_curr[i] = OrderedDict()
            self.primal_states_prev[i] = OrderedDict()

    def update(self) -> Tuple[float, float, float, float]:
        """Update global model parameters

        Return:
            Tuple[float, float, float, float]: primal residual, dual residual, minimum rho, maximum rho
        """

        # TODO: This return type may not be ideal.
        return 0.0, 0.0, 0.0, 0.0

    def get_model(self) -> nn.Module:
        """Get the model

        Return:
            nn.Module: a deepcopy of self.model
        """
        return copy.deepcopy(self.model)

    def set_weights(self, weights: OrderedDict):
        for key, value in weights.items():
            self.weights[key] = value

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

    def primal_residual_at_server(self, global_state) -> float:
        primal_res = 0
        for i in range(self.num_clients):
            for name, _ in self.model.named_parameters():
                primal_res += torch.sum(
                    torch.square(
                        global_state[name] - self.primal_states[i][name].to(self.device)
                    )
                )
        primal_res = torch.sqrt(primal_res).item()
        return primal_res

    def dual_residual_at_server(self) -> float:
        dual_res = 0
        if self.is_first_iter == 1:
            for i in range(self.num_clients):
                for name, _ in self.model.named_parameters():
                    self.primal_states_curr[i][name] = copy.deepcopy(
                        self.primal_states[i][name].to(self.device)
                    )
            self.is_first_iter = 0

        else:
            self.primal_states_prev = copy.deepcopy(self.primal_states_curr)
            for i in range(self.num_clients):
                for name, _ in self.model.named_parameters():
                    self.primal_states_curr[i][name] = copy.deepcopy(
                        self.primal_states[i][name].to(self.device)
                    )

            ## compute dual residual
            for name, _ in self.model.named_parameters():
                temp = 0
                for i in range(self.num_clients):
                    temp += self.penalty[i] * (
                        self.primal_states_prev[i][name]
                        - self.primal_states_curr[i][name]
                    )

                dual_res += torch.sum(torch.square(temp))
            dual_res = torch.sqrt(dual_res).item()

        return dual_res


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

    def __init__(
        self, id: int, weight: Dict, model: nn.Module, dataloader: DataLoader, device
    ):
        self.id = id
        self.weight = weight
        self.model = model        
        self.dataloader = dataloader
        self.device = device

        self.primal_state = OrderedDict()
        self.dual_state = OrderedDict()
        self.primal_state_curr = OrderedDict()
        self.primal_state_prev = OrderedDict()

    def update(self):
        """Update local model parameters"""
        raise NotImplementedError

    def get_model(self):
        """Get the model

        Return:
            the ``state_dict`` of local model
        """
        return self.model.state_dict()

    def primal_residual_at_client(self, global_state) -> float:
        primal_res = 0
        for name, _ in self.model.named_parameters():
            primal_res += torch.sum(
                torch.square(global_state[name] - self.primal_state[name])
            )
        primal_res = torch.sqrt(primal_res).item()
        return primal_res

    def dual_residual_at_client(self) -> float:
        dual_res = 0
        if self.is_first_iter == 1:
            self.primal_state_curr = copy.deepcopy(self.primal_state)
            self.is_first_iter = 0

        else:
            self.primal_state_prev = copy.deepcopy(self.primal_state_curr)
            self.primal_state_curr = copy.deepcopy(self.primal_state)

            ## compute dual residual
            for name, _ in self.model.named_parameters():
                temp = self.penalty * (
                    self.primal_state_prev[name] - self.primal_state_curr[name]
                )

                dual_res += torch.sum(torch.square(temp))
            dual_res = torch.sqrt(dual_res).item()

        return dual_res

    def residual_balancing(self, prim_res, dual_res):

        if prim_res > self.residual_balancing.mu * dual_res:
            self.penalty = self.penalty * self.residual_balancing.tau
        if dual_res > self.residual_balancing.mu * prim_res:
            self.penalty = self.penalty / self.residual_balancing.tau

    """ 
    Differential Privacy 
        (Laplacian mechanism) 
        - Noises from a Laplace dist. with zero mean and "scale_value" are added to the primal_state 
        - Variance = 2*(scale_value)^2
        - scale_value = sensitivty/epsilon, where sensitivity is determined by data, algorithm.
    """

    def laplace_mechanism_output_perturb(self, scale_value):
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
