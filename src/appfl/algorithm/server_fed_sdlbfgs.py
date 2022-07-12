from collections import OrderedDict
from .server_federated import FedServer

import torch
import copy


class ServerFedSDLBFGS(FedServer):

    def __init__(self, *args, **kwargs):
        super(ServerFedSDLBFGS, self).__init__(*args, **kwargs)

        """ Gradient history for L-BFGS """
        self.s_vectors = []
        self.ybar_vectors = []
        self.rho_values = []
        self.prev_params = OrderedDict()
        self.prev_grad = OrderedDict()
        self.k = 0

        # p - history
        # delta - lower bound on gamma_k
        self.p = kwargs["history"]
        self.delta = kwargs["delta"]



    def compute_step(self):
        super(ServerFedSDLBFGS, self).compute_pseudo_gradient()
        super(ServerFedSDLBFGS, self).update_m_vector()

        # Initial step, necessary so we can get x_{k - 1}
        if self.k == 0:
            self.make_initial_step()
        else:
            self.make_lbfgs_step()

            # Clean up memory
            if self.k > self.p:
                del self.s_vectors[-1]
                del self.ybar_vectors[-1]
                del self.rho_values[-1]
        self.k += 1



    def make_initial_step(self):
        for name, _ in self.model.named_parameters():
            self.step[name] = -self.pseudo_grad[name]
            self.prev_params[name] = copy.deepcopy(self.model.state_dict()[name].reshape(-1).double())
            self.prev_grad[name] = copy.deepcopy(self.pseudo_grad[name].reshape(-1).double())


    def make_lbfgs_step(self):

        self.s_vectors.insert(0, OrderedDict())
        self.ybar_vectors.insert(0, OrderedDict())
        self.rho_values.insert(0, OrderedDict())

        for name, _ in self.model.named_parameters():

            shape = self.model.state_dict()[name].shape

            # Create newest s vector
            s_vector = self.model.state_dict()[name].reshape(-1).double() - self.prev_params[name]

            self.s_vectors[0][name] = s_vector

            # Create newest ybar vector
            y_vector = self.pseudo_grad[name].reshape(-1).double() - self.prev_grad[name]
            gamma = self.compute_gamma(y_vector, s_vector)
            ybar_vector = self.compute_ybar_vector(y_vector, s_vector, gamma)
            self.ybar_vectors[0][name] = ybar_vector

            # Create newest rho
            rho = 1.0 / (s_vector.dot(ybar_vector))
            self.rho_values[0][name] = rho

            # Perform recursive computations and step
            v_vector = self.compute_step_approximation(name, gamma)

            self.step[name] = -(self.server_learning_rate * v_vector.reshape(shape))

            # Store information for next step
            self.prev_params[name] = copy.deepcopy(self.model.state_dict()[name].reshape(-1).double())
            self.prev_grad[name] = copy.deepcopy(v_vector)



    def compute_gamma(self, y_vec, s_vec):
        return max((y_vec.dot(y_vec)) / (y_vec.dot(s_vec)), self.delta)


    def compute_theta(self, y_vec, s_vec, gamma):
        s_proj = s_vec.dot(s_vec) * gamma
        dot = s_vec.dot(y_vec)

        if dot >= 0.25 * s_proj:
            return 1
        return (0.75 * s_proj) / (s_proj - dot)


    def compute_ybar_vector(self, y_vec, s_vec, gamma):

        # H_{k, 0}^{-1} doesn't need to be computed directly since 
        # H_{k, 0}^{-1} @ vector = (gamma * I) @ vector = gamma * vector
        theta = self.compute_theta(y_vec, s_vec, gamma)
        val = (theta * y_vec) + ((1 - theta) * (gamma * s_vec))

        return val


    def compute_step_approximation(self, name, gamma):
        u = self.pseudo_grad[name].reshape(-1).double()
        mu_values = []
        m = min(self.p, self.k - 1)
        r = range(m)

        for i in r:
            mu = self.rho_values[i][name] * u.dot(self.s_vectors[i][name])
            mu_values.insert(0, mu)
            u = u - (mu * self.ybar_vectors[i][name])

        v = (1.0 / gamma) * u
        for i in reversed(r):
            nu = self.rho_values[i][name] * v.dot(self.ybar_vectors[i][name])
            v = v + (mu_values[m - i - 1] - nu) * self.s_vectors[i][name]
        
        return v

    def logging_summary(self, cfg, logger):
        super(FedServer, self).log_summary(cfg, logger)

        logger.info("client_learning_rate = %s " % (cfg.fed.args.optim_args.lr))

        if cfg.summary_file != "":
            with open(cfg.summary_file, "a") as f:

                f.write(
                    cfg.logginginfo.DataSet_name
                    + " FedSDLBFGS ClientLR "
                    + str(cfg.fed.args.optim_args.lr)
                    + " TestAccuracy "
                    + str(cfg.logginginfo.accuracy)
                    + " BestAccuracy "
                    + str(cfg.logginginfo.BestAccuracy)
                    + " Time "
                    + str(round(cfg.logginginfo.Elapsed_time, 2))
                    + "\n"
                )
