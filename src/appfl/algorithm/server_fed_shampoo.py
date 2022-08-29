from collections import OrderedDict
from .server_federated import FedServer

import torch
from torch import linalg
import copy

from ..misc.utils import validation


class ServerFedShampoo(FedServer):

    def __init__(self, *args, **kwargs):
        super(ServerFedShampoo, self).__init__(*args, **kwargs)

        self.state = OrderedDict()
        for name, _ in self.model.named_parameters():
            self.state[name] = OrderedDict()

        self.momentum = kwargs["server_momentum_param_2"]
        self.weight_decay = kwargs["weight_decay"]
        self.epsilon = kwargs["second_order_epsilon"]
        self.update_freq = kwargs["update_freq"]


    def _matrix_power(self, matrix, power):
        # use CPU for svd for speed up
        # matrix = matrix.cpu()
        u, s, v = torch.svd(matrix)
        return (u @ s.pow_(power).diag() @ v.t()).cuda()

    def compute_step(self):
        super().compute_pseudo_gradient()
        for name, _ in self.model.named_parameters():

            p = self.model.state_dict()[name]
            grad = -self.pseudo_grad[name]
            order = grad.ndimension()
            original_size = grad.size()

            # __import__('pdb').set_trace()
            state = self.state[name]

            if len(state) == 0:
                state["step"] = 0
                if self.momentum > 0:
                    state["momentum_buffer"] = grad.clone()
                for dim_id, dim in enumerate(grad.size()):
                    # precondition matrices
                    state[f"precond_{dim_id}"] = self.epsilon * torch.eye(dim, out=grad.new(dim, dim))
                    state[f"precond_{dim_id}"] = torch.eye(dim, out=grad.new(dim, dim))
                    state[f"inv_precond_{dim_id}"] = grad.new(dim, dim).zero_()

            if self.momentum > 0:
                # grad = (1 - moment) * grad(t) + moment * grad(t-1)
                # and grad(-1) = grad(0)
                grad.mul_(1 - self.momentum).add_(self.momentum, state["momentum_buffer"])

            if self.weight_decay > 0:
                grad.add_(self.weight_decay, p.data)

            # See Algorithm 2 for detail
            for dim_id, dim in enumerate(grad.size()):
                precond = state[f"precond_{dim_id}"]
                inv_precond = state[f"inv_precond_{dim_id}"]

                # mat_{dim_id}(grad)
                grad = grad.transpose_(0, dim_id).contiguous()
                transposed_size = grad.size()
                grad = grad.view(dim, -1)

                grad_t = grad.t()
                precond.add_(grad @ grad_t)
                if state["step"] % self.update_freq == 0:
                    inv_precond.copy_(self._matrix_power(precond, -1 / order))
                inv_precond.copy_(self._matrix_power(precond, -1 / order))

                if dim_id == order - 1:
                    # finally
                    grad = grad_t @ inv_precond
                    # grad: (-1, last_dim)
                    grad = grad.view(original_size)
                else:
                    # if not final
                    grad = inv_precond @ grad
                    # grad (dim, -1)
                    grad = grad.view(transposed_size)

            state["step"] += 1
            state["momentum_buffer"] = grad

            self.step[name] = -self.server_learning_rate * grad




    def logging_summary(self, cfg, logger):
        super(FedServer, self).log_summary(cfg, logger)

        logger.info("client_learning_rate = %s " % (cfg.fed.args.optim_args.lr))

        if cfg.summary_file != "":
            with open(cfg.summary_file, "a") as f:

                f.write(
                    cfg.logginginfo.DataSet_name
                    + " FedShampoo ClientLR "
                    + str(cfg.fed.args.optim_args.lr)
                    + " TestAccuracy "
                    + str(cfg.logginginfo.accuracy)
                    + " BestAccuracy "
                    + str(cfg.logginginfo.BestAccuracy)
                    + " Time "
                    + str(round(cfg.logginginfo.Elapsed_time, 2))
                    + "\n"
                )
