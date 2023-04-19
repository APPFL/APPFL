import logging

from collections import OrderedDict
from .server_federated import FedServer
from ..misc import *

logger = logging.getLogger(__name__)

class ServerFedAsynchronous(FedServer):
    """ Implement FedAsync algorithm
        Asynchronous Federated Optimization: http://arxiv.org/abs/1903.03934
    
    Agruments:
        weights: weight for each client
        model (nn.Module): PyTorch model
        loss_fn (nn.Module): loss function
        num_clients (int): number of clients
        device (str): server's device for running evaluation  
    """
    def __init__(self, weights, model, loss_fn, num_clients, device, global_step = 0, staness_func = 'constant', **kwargs):
        weights = [1.0 / num_clients for _ in range(num_clients)] if weights is None else weights
        self.global_step = global_step
        # Create staleness function (Sec. 5.2) 
        self.staleness = self.__staleness_func_factory(
            stalness_func_name= staness_func['name'],
            **staness_func['args']
        )
        super(ServerFedAsynchronous, self).__init__(weights, model, loss_fn, num_clients, device, **kwargs)

    def compute_pseudo_gradient(self, clinet_idx):
        for name, _ in self.model.named_parameters():
            self.pseudo_grad[name] = torch.zeros_like(self.model.state_dict()[name])
            # TODO: Check the use of weights here
            # self.pseudo_grad[name] += (self.global_state[name]-self.primal_states[clinet_idx][name])
            self.pseudo_grad[name] += self.weights[clinet_idx] * (self.global_state[name]-self.primal_states[clinet_idx][name])

    def compute_step(self, init_step: int, client_idx: int, E_weight: float):
        self.compute_pseudo_gradient(client_idx)
        for name, _ in self.model.named_parameters():
            # Apply staleness factor
            alpha_t = self.alpha * self.staleness(self.global_step - init_step)
            self.step[name] = - alpha_t * self.pseudo_grad[name] * E_weight

    def primal_residual_at_server(self, client_idx: int) -> float:
        primal_res = 0
        for name, _ in self.model.named_parameters():
            primal_res += torch.sum(torch.square(self.global_state[name]-self.primal_states[client_idx][name].to(self.device)))
        self.prim_res = torch.sqrt(primal_res).item()

    def update(self, local_states: OrderedDict, init_step: int, client_idx: int, E_weight: float):  
        # Obtain the global and local states
        self.global_state = copy.deepcopy(self.model.state_dict())
        super(FedServer, self).primal_recover_from_local_states(local_states)
        # Calculate residual
        self.primal_residual_at_server(client_idx)
        # Change device
        for name, _ in self.model.named_parameters():
            self.primal_states[client_idx][name] = self.primal_states[client_idx][name].to(self.device)
        # Global state computation
        self.compute_step(init_step, client_idx, E_weight)
        for name, _ in self.model.named_parameters():
            self.global_state[name] += self.step[name]
        # Model update
        self.model.load_state_dict(self.global_state)
        # Global step update
        self.global_step += 1

    def logging_summary(self, cfg, logger):
        super(FedServer, self).log_summary(cfg, logger)

    def __staleness_func_factory(self, stalness_func_name, **kwargs):
        if stalness_func_name   == "constant":
            return lambda u : 1
        elif stalness_func_name == "polynomial":
            a = kwargs['a']
            return lambda u:  (u + 1) ** a
        elif stalness_func_name == "hinge":
            a = kwargs['a']
            b = kwargs['b']
            return lambda u: 1 if u <= b else 1.0/ (a * (u - b) + 1.0)
        else:
            raise NotImplementedError
