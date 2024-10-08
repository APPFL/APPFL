import logging
import copy
from .server_federated import FedServer
from appfl.misc import *

logger = logging.getLogger(__name__)

@deprecated("Imports from appfl.algorithm is deprecated and will be removed in the future. Please use appfl.algorithm.aggregator instead.")
class ServerAREA(FedServer):
    """ 
    ServerAREA
        Asynchronous Federated Stochastic Optimization for Heterogeneous Objectives Under Arbitrary Delays: https://arxiv.org/abs/2405.10123
    Args:
        weights: weight for each client
        model: PyTorch model
        loss_fn: loss function
        num_clients: number of clients
        device: server device
    """
    def __init__(self, weights, model, loss_fn, num_clients, device, **kwargs):
        self.counter = 0 
        self.global_step = 0
        weights = [1.0 / num_clients for _ in range(num_clients)] if weights is None else weights 
        super(ServerAREA, self).__init__(weights, model, loss_fn, num_clients, device, **kwargs)

        # Use pseudo_grad variable to store aggregator
        for name in self.model.state_dict():
            self.pseudo_grad[name] = torch.zeros_like(self.model.state_dict()[name])
        self.list_named_parameters = []
        for name, _ in self.model.named_parameters():
            self.list_named_parameters.append(name)

    def update_aggregator(self, local_gradient: dict):
        for name in self.model.state_dict():
            self.pseudo_grad[name] += local_gradient[name]
        self.counter += 1

    def update(self, local_model: dict, **kwargs): 
        self.update_aggregator(local_model)
        if self.counter == self.K:
            self.global_state = copy.deepcopy(self.model.state_dict())
            for name in self.model.state_dict():
                self.global_state[name] += self.pseudo_grad[name] / self.num_clients
            self.model.load_state_dict(self.global_state)
            self.global_step += 1
            self.counter = 0
            for name in self.model.state_dict():
                self.pseudo_grad[name] = torch.zeros_like(self.model.state_dict()[name])
        
    def logging_summary(self, cfg, logger):
        super(FedServer, self).log_summary(cfg, logger)
        logger.info("client_learning_rate=%s " % (cfg.fed.args.optim_args.lr))
        logger.info("buffer_size=%s" % (cfg.fed.args.K))
        if cfg.summary_file != "":
            with open(cfg.summary_file, "a") as f:
                f.write(
                    cfg.logginginfo.DataSet_name
                    + " AREA ClientLR "
                    + str(cfg.fed.args.optim_args.lr)
                    + " AREA Buffer Size"
                    + str(cfg.fed.args.K)
                    + " TestAccuracy "
                    + str(cfg.logginginfo.accuracy)
                    + " BestAccuracy "
                    + str(cfg.logginginfo.BestAccuracy)
                    + " Time "
                    + str(round(cfg.logginginfo.Elapsed_time, 2))
                    + "\n"
                )
