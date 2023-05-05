import logging
import copy
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
    def __init__(self, weights, model, loss_fn, num_clients, device, **kwargs):
        weights = [1.0 / num_clients for _ in range(num_clients)] if weights is None else weights
        super(ServerFedAsynchronous, self).__init__(weights, model, loss_fn, num_clients, device, **kwargs)
        self.global_step = 0
        self.staleness = self.__staleness_func_factory(
            stalness_func_name= self.staleness_func['name'],
            **self.staleness_func['args']
        )

    def compute_pseudo_gradient(self, local_state: dict, client_idx: int):
        for name, _ in self.model.named_parameters():
            self.pseudo_grad[name] = torch.zeros_like(self.model.state_dict()[name])
            self.pseudo_grad[name] += self.weights[client_idx] * (self.global_state[name] - local_state[name])

    def compute_step(self, local_state: dict, init_step: int, client_idx: int):
        self.compute_pseudo_gradient(local_state, client_idx)
        for name, _ in self.model.named_parameters():
            alpha_t = self.alpha * self.staleness(self.global_step - init_step)
            self.step[name] = - alpha_t * self.pseudo_grad[name]

    def update(self, local_state: dict, init_step: int, client_idx: int):  
        self.global_state = copy.deepcopy(self.model.state_dict())
        self.compute_step(local_state, init_step, client_idx)
        for name, _ in self.model.named_parameters():
            self.global_state[name] += self.step[name]
        self.model.load_state_dict(self.global_state)
        self.global_step += 1

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
        
    def logging_summary(self, cfg, logger):
        super(FedServer, self).log_summary(cfg, logger)
        logger.info("client_learning_rate=%s " % (cfg.fed.args.optim_args.lr))
        logger.info("model_mixing_parameter=%s " % (cfg.fed.args.alpha))
        logger.info("staleness_func=%s" % (cfg.fed.args.staleness_func.name))

        if cfg.summary_file != "":
            with open(cfg.summary_file, "a") as f:

                f.write(
                    cfg.logginginfo.DataSet_name
                    + " FedAsync ClientLR "
                    + str(cfg.fed.args.optim_args.lr)
                    + " FedAsync Alpha "
                    + str(cfg.fed.args.alpha)
                    + " FedAsync Staleness Function"
                    + str(cfg.fed.args.staleness_func.name)
                    + " TestAccuracy "
                    + str(cfg.logginginfo.accuracy)
                    + " BestAccuracy "
                    + str(cfg.logginginfo.BestAccuracy)
                    + " Time "
                    + str(round(cfg.logginginfo.Elapsed_time, 2))
                    + "\n"
                )
