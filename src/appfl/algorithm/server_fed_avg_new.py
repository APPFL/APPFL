import copy
import torch
from torch.optim import *
from collections import OrderedDict
from .server_federated import FedServer

class ServerFedAvgNew(FedServer):
    def update(self, local_states: OrderedDict):
        """Inputs for the global model update"""
        self.global_state = copy.deepcopy(self.model.state_dict())
        for name, _ in self.model.named_parameters():
            self.step[name] = torch.zeros_like(self.model.state_dict()[name])
            for sid, states in enumerate(local_states):
                if states is not None:
                    if self.gradient_based:
                        self.step[name] -= self.weights[sid] * states[name]
                    else:
                        self.step[name] -= self.weights[sid] * (self.global_state[name] - states[name])

        """ global_state calculation """
        for name in self.model.state_dict():    
            self.global_state[name] += self.step[name]      
                

        """ model update """
        self.model.load_state_dict(self.global_state)


    def logging_summary(self, cfg, logger):
        super(FedServer, self).log_summary(cfg, logger)

        logger.info("client_learning_rate = %s " % (cfg.fed.args.optim_args.lr))

        if cfg.summary_file != "":
            with open(cfg.summary_file, "a") as f:

                f.write(
                    cfg.logginginfo.DataSet_name
                    + " FedAvg ClientLR "
                    + str(cfg.fed.args.optim_args.lr)
                    + " TestAccuracy "
                    + str(cfg.logginginfo.accuracy)
                    + " BestAccuracy "
                    + str(cfg.logginginfo.BestAccuracy)
                    + " Time "
                    + str(round(cfg.logginginfo.Elapsed_time, 2))
                    + "\n"
                )
