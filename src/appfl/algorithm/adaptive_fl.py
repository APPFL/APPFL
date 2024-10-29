from .fl_base import BaseServer, BaseClient
from .server_federated import FedServer
from collections import OrderedDict
from torch.optim import *
import torch
import copy
import logging

class AdaptiveFLServer(FedServer):
    def __init__(self, weights, model, loss_fn, num_clients, device, server_lr =1e-3, gamma=1.4, **kwargs):
        super(AdaptiveFLServer, self).__init__(weights, model, loss_fn, num_clients, device)
        self.server_lr = server_lr
        self.gamma = gamma
        self.lr_clients = {client_id: server_lr for client_id in range(num_clients)}  # Initialize learning rates for all clients
        self.__dict__.update(kwargs)

    def update(self, local_states):
        gradients = OrderedDict()
        func_val_diffs = OrderedDict()
        # for k in range(self.num_clients):
        #     print(f"Server updated learning rate for client {k}: {self.lr_clients[k]}")

        for client_id, state in enumerate(local_states):
            gradients[client_id] = {k: v.to(self.device) for k, v in state['grad_estimate'].items()}
            func_val_diffs[client_id] = torch.tensor(state['function_value_difference'], device=self.device)

        total_func_val_diff = sum(func_val_diffs.values())
        selected_clients = []

        for client_id in range(self.num_clients):
            gradient_norm = torch.norm(torch.cat([gradients[client_id][k].view(-1) for k in gradients[client_id]])).item()
            if func_val_diffs[client_id] <= -self.lr_clients[client_id] * gradient_norm ** 2:
                selected_clients.append(client_id)
        #     print("func_val_diffs: ",func_val_diffs[client_id])
        #     print("learning rate: ", self.lr_clients[client_id])
        #     print("gradient norm: ",gradient_norm)
            print("LHS : ", round(func_val_diffs[client_id].item(), 4)," VS ","RHS: ",round(-self.lr_clients[client_id] * gradient_norm ** 2,4),"lr_c_k: ",self.lr_clients[client_id],"grad_norm: ",  gradient_norm)
            # print("RHS value (lr * grad_norm): ",round(-self.lr_clients[client_id] * gradient_norm ** 2,2))
        print(f"Selected Clients for Global Update: {selected_clients}")

        global_state = copy.deepcopy(self.model.state_dict())
        global_state = {k: v.to(self.device) for k, v in global_state.items()}

        if selected_clients:
            for key in global_state.keys():
                global_state[key] = torch.zeros_like(global_state[key], device=self.device)
                for client_id in selected_clients:
                    local_param = local_states[client_id]['primal_state'][key].to(self.device)
                    global_state[key] += local_param * self.weights[client_id]

                global_state[key] /= len(selected_clients)
   
            self.model.load_state_dict(global_state)

        # Update learning rates for clients
        for k in range(self.num_clients):
            if k in selected_clients:
                self.lr_clients[k] *= self.gamma
            else:
                self.lr_clients[k] /= self.gamma

        # Return both global state and learning rates separately
        print(f"Server sending learning rate to client {k}: {self.lr_clients[k]}")
        # stop
        return global_state, self.lr_clients


    def logging_iteration(self, cfg, logger, t):
        if t == 0:
            title = super(AdaptiveFLServer, self).log_title()
            logger.info(title)
        contents = super(AdaptiveFLServer, self).log_contents(cfg, t)
        logger.info(contents)

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