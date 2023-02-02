from .server_federated import FedServer
from collections import OrderedDict
import copy
import torch

class ServerFedSDLBFGS(FedServer):

    def __init__(self, *args, **kwargs):
        super(ServerFedSDLBFGS, self).__init__(*args, **kwargs)

        # p - history
        # delta - lower bound on gamma_k
        self.p = kwargs["history"]
        self.delta = kwargs["delta"] 

        """ Gradient history for L-BFGS """
        self.s_vectors = [] 
        self.ybar_vectors = []
        self.rho_values = []
        self.k = 0

        self.prim_avg=OrderedDict()
        self.yvec_avg=OrderedDict()
        self.svec_avg=OrderedDict()
        self.grad_avg=OrderedDict()


    def update(self, local_states: OrderedDict):
        
        
        """Inputs for the global model update"""        
        super(ServerFedSDLBFGS, self).primal_recover_from_local_states(local_states)
        super(ServerFedSDLBFGS, self).yvec_recover_from_local_states(local_states)        
        super(ServerFedSDLBFGS, self).svec_recover_from_local_states(local_states)
        super(ServerFedSDLBFGS, self).grad_recover_from_local_states(local_states)
         
 
        """ change device """
        for i in range(self.num_clients): 
            for name in self.model.state_dict():
                self.primal_states[i][name] = self.primal_states[i][name].to(
                    self.device
                )
            for name, p in self.model.named_parameters():                
                self.yvecs[i][name] = self.yvecs[i][name].to(
                    self.device
                )
                self.svecs[i][name] = self.svecs[i][name].to(
                    self.device
                )
                self.grads[i][name] = self.grads[i][name].to(
                    self.device
                )
        """ FedAvg """                          
        for name in self.model.state_dict():
            self.prim_avg[name] = torch.zeros_like(self.model.state_dict()[name], device=self.device)
 
            if self.prim_avg[name].ndim > 0: 
                for i in range(self.num_clients):                
                    self.prim_avg[name] += self.weights[i] * self.primal_states[i][name]          
                 
        """ model update """
        self.model.load_state_dict(self.prim_avg)       


        """ SdLBFGS """
        ## Averaging
        for name, p in self.model.named_parameters():   
            self.yvec_avg[name] = torch.zeros_like(p.data.reshape(-1), device=self.device)
            self.svec_avg[name] = torch.zeros_like(p.data.reshape(-1), device=self.device)
            self.grad_avg[name] = torch.zeros_like(p.data.reshape(-1), device=self.device)
            for i in range(self.num_clients): 
                self.svec_avg[name] += self.svecs[i][name]
                self.yvec_avg[name] += self.yvecs[i][name]
                self.grad_avg[name] += self.grads[i][name]
            self.svec_avg[name] = self.svec_avg[name]/self.num_clients
            self.yvec_avg[name] = self.yvec_avg[name]/self.num_clients
            self.grad_avg[name] = self.grad_avg[name]/self.num_clients

 
        self.make_lbfgs_step()


 
    def make_lbfgs_step(self):
        optimizer = torch.optim.SGD(self.model.parameters(),lr=self.server_learning_rate) 

        self.s_vectors.append(OrderedDict())
        self.ybar_vectors.append(OrderedDict())
        self.rho_values.append(OrderedDict())

        if self.k > self.p:
            del self.s_vectors[0]
            del self.ybar_vectors[0]
            del self.rho_values[0]

        for name, p in self.model.named_parameters():

            shape = self.model.state_dict()[name].shape

            # Create newest s vector
            s_vector =  self.svec_avg[name]
            self.s_vectors[-1][name] = s_vector

            # Create newest ybar vector
            y_vector = self.yvec_avg[name]
            gamma = self.compute_gamma(y_vector, s_vector)
            ybar_vector = self.compute_ybar_vector(y_vector, s_vector, gamma)
            self.ybar_vectors[-1][name] = ybar_vector

            # Create newest rho
            rho = 1.0 / (s_vector.dot(ybar_vector))
            self.rho_values[-1][name] = rho

            # Perform recursive computations and step
            v_vector = self.compute_step_approximation(name, gamma)              
            p.grad = v_vector.reshape(shape)    
             
        optimizer.step()
 

    def compute_gamma(self, y_vec, s_vec):        
        return max((y_vec.dot(y_vec)) / (y_vec.dot(s_vec)), self.delta)


    def compute_theta(self, y_vec, s_vec, gamma):        
        s_proj = s_vec.dot(s_vec) * gamma
        dot = s_vec.dot(y_vec)

        if dot < 0.25 * s_proj:
            return (0.75 * s_proj) / (s_proj - dot)
        return 1


    def compute_ybar_vector(self, y_vec, s_vec, gamma):
 
        theta = self.compute_theta(y_vec, s_vec, gamma)
        val = (theta * y_vec) + ((1 - theta) * (gamma * s_vec))

        return val
 
    def compute_step_approximation(self, name, gamma):
        
        u = copy.deepcopy(self.grad_avg[name])
        mu_values = []
        m = min(self.k - 1, self.p)
        r = range(m)

        for i in reversed(r):
            mu = self.rho_values[i][name] * u.dot(self.s_vectors[i][name])
            mu_values.append(mu)
            u = u - (mu * self.ybar_vectors[i][name])
             
        v = (1.0 / gamma) * u
        for i in r:
            nu = self.rho_values[i][name] * v.dot(self.ybar_vectors[i][name])
            v = v + ((mu_values[-(i + 1)] - nu) * self.s_vectors[i][name])
              
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