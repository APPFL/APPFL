import logging
import copy
from collections import OrderedDict
from .server_federated import FedServer
from ..misc import *

logger = logging.getLogger(__name__)

class ServerFedCPASAvg(FedServer):
    """ Implement FedBuffer algorithm
        Asynchronous Federated Optimization: https://arxiv.org/abs/2106.06639
    
    Agruments:
        weights: weight for each client
        model (nn.Module): PyTorch model
        loss_fn (nn.Module): loss function
        num_clients (int): number of clients
        device (str): server's device for running evaluation  
    """
    def __init__(self, weights, model, loss_fn, num_clients, device, **kwargs):
        self.counter = 0 
        self.global_step = 0
        weights = [1.0 / num_clients for _ in range(num_clients)] if weights is None else weights
        super(ServerFedCPASAvg, self).__init__(weights, model, loss_fn, num_clients, device, **kwargs)
        self.staleness = self.__staleness_func_factory(
            stalness_func_name= self.staleness_func['name'],
            **self.staleness_func['args']
        )
        self.group_pseudo_grad = OrderedDict()

    def update(self, local_gradient: dict, init_step: int, client_idx: int):
        """Update the model directly using the client local gradient with staleness factor applied."""
        self.global_state = copy.deepcopy(self.model.state_dict())
        alpha_t = self.alpha * self.staleness(self.global_step - init_step)
        for name, _ in self.model.named_parameters():
            self.global_state[name] -= local_gradient[name] * self.weights[client_idx] * alpha_t
        self.model.load_state_dict(self.global_state)
        self.global_step += 1

    def buffer(self, local_gradient: dict, init_step: int, client_idx: int, group_idx: int):
        """Buffer the local gradient from the client of a certain group."""
        if group_idx not in self.group_pseudo_grad:
            self.group_pseudo_grad[group_idx] = OrderedDict()
            for name, _ in self.model.named_parameters():
                self.group_pseudo_grad[group_idx][name] = torch.zeros_like(self.model.state_dict()[name])
        alpha_t = self.alpha * self.staleness(self.global_step - init_step)
        for name, _ in self.model.named_parameters():
            self.group_pseudo_grad[group_idx][name] += local_gradient[name] * self.weights[client_idx] * alpha_t
        
    def update_group(self, group_idx: int):
        """Update the model using all the buffered gradients for a certain group."""
        if group_idx in self.group_pseudo_grad:
            self.global_state = copy.deepcopy(self.model.state_dict())
            for name, _ in self.model.named_parameters():
                self.global_state[name] -= self.group_pseudo_grad[group_idx][name]
            self.model.load_state_dict(self.global_state)
            self.global_step += 1
            del self.group_pseudo_grad[group_idx]

    def update_all(self):
        """Update the model using all the buffered gradients for a group."""
        self.global_state = copy.deepcopy(self.model.state_dict())
        for group_idx in self.group_pseudo_grad:
            for name, _ in self.model.named_parameters():
                self.global_state[name] -= self.group_pseudo_grad[group_idx][name]
        self.group_pseudo_grad = OrderedDict()
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
                    + " ServerFedCPASAvg ClientLR "
                    + str(cfg.fed.args.optim_args.lr)
                    + " ServerFedCPASAvg Alpha "
                    + str(cfg.fed.args.alpha)
                    + " ServerFedCPASAvg Staleness Function"
                    + str(cfg.fed.args.staleness_func.name)
                    + " TestAccuracy "
                    + str(cfg.logginginfo.accuracy)
                    + " BestAccuracy "
                    + str(cfg.logginginfo.BestAccuracy)
                    + " Time "
                    + str(round(cfg.logginginfo.Elapsed_time, 2))
                    + "\n"
                )
