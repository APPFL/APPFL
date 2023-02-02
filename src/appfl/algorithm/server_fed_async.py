import logging

from appfl.algorithm.server_async_federated import AsyncFedServer
from appfl.algorithm.server_federated import FedServer

logger = logging.getLogger(__name__)


class ServerFedAsync(AsyncFedServer):
    """Implement FedAsync algorithm
        Asynchronous Federated Optimization: http://arxiv.org/abs/1903.03934

    Agruments:
        weights: weight for each client
        model (nn.Module): PyTorch model
        loss_fn (nn.Module): loss function
        num_clients (int): number of clients
        device (str): server's device for running evaluation
    """

    def __init__(
        self,
        model,
        loss_fn,
        num_clients,
        device,
        global_step=0,
        staness_func="constant",
        weights=None,
        **kwargs
    ):
        # FedAsync does not apply any weighting for clients
        weights = [1.0 for i in range(num_clients)] if weights is None else weights
        super(ServerFedAsync, self).__init__(
            weights, model, loss_fn, num_clients, device, global_step, **kwargs
        )

        # Create staleness function (Sec. 5.2)
        self.staleness = self.__staleness_func_factory(
            stalness_func_name=staness_func["name"], **staness_func["args"]
        )

    def compute_step(self, init_step: int):
        super(ServerFedAsync, self).compute_pseudo_gradient()
        for name, _ in self.model.named_parameters():
            # Apply staleness factor
            alpha_t = self.alpha * self.staleness(self.global_step - init_step)
            self.step[name] = -alpha_t * self.pseudo_grad[name]

    def logging_summary(self, cfg, logger):
        super(FedServer, self).log_summary(cfg, logger)

    def __staleness_func_factory(self, stalness_func_name, **kwargs):
        if stalness_func_name == "constant":
            return lambda u: 1
        elif stalness_func_name == "polynomial":
            a = kwargs["a"]
            return lambda u: (u + 1) ** a
        elif stalness_func_name == "hinge":
            a = kwargs["a"]
            b = kwargs["b"]
            return lambda u: 1 if u <= b else 1.0 / (a * (u - b) + 1.0)
        else:
            raise NotImplementedError
