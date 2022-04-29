import copy
from typing import Dict, Tuple

from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig

import os
import logging
import numpy as np
from sklearn.decomposition import PCA


class BaseServer:
    """Abstract class of PPFL algorithm for server that aggregates and updates model parameters.

    Args:
        weight (Dict): aggregation weight assigned to each client
        model (nn.Module): torch neural network model to train
        num_clients (int): the number of clients
        device (str): device for computation
    """

    def __init__(
        self, weights: OrderedDict, model: nn.Module, num_clients: int, device
    ):
        self.model = model
        self.num_clients = num_clients
        self.device = device
        self.weights = copy.deepcopy(weights)
        self.penalty = OrderedDict()
        self.prim_res = 0
        self.dual_res = 0
        self.reduced_grad_vec = OrderedDict()
        self.param_vec = OrderedDict()
        self.global_state = OrderedDict()
        self.primal_states = OrderedDict()
        self.dual_states = OrderedDict()
        self.primal_states_curr = OrderedDict()
        self.primal_states_prev = OrderedDict()
        for i in range(num_clients):
            self.primal_states[i] = OrderedDict()
            self.dual_states[i] = OrderedDict()
            self.primal_states_curr[i] = OrderedDict()
            self.primal_states_prev[i] = OrderedDict()

    def get_model(self) -> nn.Module:
        """Get the model

        Return:
            nn.Module: a deepcopy of self.model
        """
        return copy.deepcopy(self.model)

    def set_weights(self, weights: OrderedDict):
        for key, value in weights.items():
            self.weights[key] = value

    def grad_vec_recover_from_local_states(self, local_states):
        for _, states in enumerate(local_states):
            if states is not None:
                for sid, state in states.items():
                    self.reduced_grad_vec[sid] = copy.deepcopy(state["grad"])

    def param_vec_recover_from_local_states(self, local_states):
        for _, states in enumerate(local_states):
            if states is not None:
                for sid, state in states.items():
                    self.param_vec[sid] = copy.deepcopy(state["param_vec"])

    def primal_recover_from_local_states(self, local_states):
        for _, states in enumerate(local_states):
            if states is not None:
                for sid, state in states.items():
                    self.primal_states[sid] = copy.deepcopy(state["primal"])

    def dual_recover_from_local_states(self, local_states):
        for _, states in enumerate(local_states):
            if states is not None:
                for sid, state in states.items():
                    self.dual_states[sid] = copy.deepcopy(state["dual"])

    def penalty_recover_from_local_states(self, local_states):
        for _, states in enumerate(local_states):
            if states is not None:
                for sid, state in states.items():
                    self.penalty[sid] = copy.deepcopy(state["penalty"][sid])

    def primal_residual_at_server(self) -> float:
        primal_res = 0
        for i in range(self.num_clients):
            for name, _ in self.model.named_parameters():
                primal_res += torch.sum(
                    torch.square(
                        self.global_state[name]
                        - self.primal_states[i][name].to(self.device)
                    )
                )
        self.prim_res = torch.sqrt(primal_res).item()

    def dual_residual_at_server(self) -> float:
        dual_res = 0
        if self.is_first_iter == 1:
            for i in range(self.num_clients):
                for name, _ in self.model.named_parameters():
                    self.primal_states_curr[i][name] = copy.deepcopy(
                        self.primal_states[i][name].to(self.device)
                    )
            self.is_first_iter = 0

        else:
            self.primal_states_prev = copy.deepcopy(self.primal_states_curr)
            for i in range(self.num_clients):
                for name, _ in self.model.named_parameters():
                    self.primal_states_curr[i][name] = copy.deepcopy(
                        self.primal_states[i][name].to(self.device)
                    )

            ## compute dual residual
            for name, _ in self.model.named_parameters():
                temp = 0
                for i in range(self.num_clients):
                    temp += self.penalty[i] * (
                        self.primal_states_prev[i][name]
                        - self.primal_states_curr[i][name]
                    )

                dual_res += torch.sum(torch.square(temp))
            self.dual_res = torch.sqrt(dual_res).item()

    def log_title(self):
        title = "%10s %10s %10s %10s %10s %10s %10s %10s" % (
            "Iter",
            "Local(s)",
            "Global(s)",
            "Valid(s)",
            "PerIter(s)",
            "Elapsed(s)",
            "TestLoss",
            "TestAccu",
        )
        return title

    def log_contents(self, cfg, t):
        contents = "%10d %10.2f %10.2f %10.2f %10.2f %10.2f %10.4f %10.2f" % (
            t + 1,
            cfg["logginginfo"]["LocalUpdate_time"],
            cfg["logginginfo"]["GlobalUpdate_time"],
            cfg["logginginfo"]["Validation_time"],
            cfg["logginginfo"]["PerIter_time"],
            cfg["logginginfo"]["Elapsed_time"],
            cfg["logginginfo"]["test_loss"],
            cfg["logginginfo"]["test_accuracy"],
        )
        return contents

    def log_summary(self, cfg: DictConfig, logger):

        logger.info("Device=%s" % (cfg.device))
        logger.info("#Processors=%s" % (cfg["logginginfo"]["comm_size"]))
        logger.info("#Clients=%s" % (self.num_clients))
        logger.info("Server=%s" % (cfg.fed.servername))
        logger.info("Clients=%s" % (cfg.fed.clientname))
        logger.info("Comm_Rounds=%s" % (cfg.num_epochs))
        logger.info("Local_Rounds=%s" % (cfg.fed.args.num_local_epochs))
        logger.info("DP_Eps=%s" % (cfg.fed.args.epsilon))
        logger.info("Clipping=%s" % (cfg.fed.args.clip_value))
        logger.info("Elapsed_time=%s" % (round(cfg["logginginfo"]["Elapsed_time"], 2)))
        logger.info("BestAccuracy=%s" % (round(cfg["logginginfo"]["BestAccuracy"], 2)))



    """ 
    Projection Matrix
        Each client can obtain a trajectory of iterates by training a model using the local data, i.e., w^1, w^2, ..., w^I, where w^i in R^J
        By conduting PCA (principle component analysis) using the "I" iterates, one can obtain a projection matrix "M X J" where M ( < I ) is the number of components.
    """


    def log_pca(self):

        self.outfile.write("pca_dir: %s \n" %(self.pca_dir) )
        self.outfile.write("Ratio: %s \n" % (self.EVR))
        self.outfile.write("Sum: %s \n" % ( sum(self.EVR) ) )        
        self.outfile.write("P: (%s, %s) \n" % (self.P.shape[0], self.P.shape[1])) 
        self.outfile.flush() 

    def construct_projection_matrix(self):
  
        W = []
 
        for i in range(self.params_start, self.params_end):
            self.model.load_state_dict(
                torch.load(
                    os.path.join(self.pca_dir , "%s.pt" %(i) ),
                    map_location=torch.device(self.device),
                )
            )
            W.append(self.get_model_param_vec())

        W = np.array(W)
        
        # Obtain base variables through PCA
        pca = PCA(n_components=self.ncomponents)
        pca.fit_transform(W)
        P = np.array(pca.components_)
        P = torch.from_numpy(P).to(self.device)

        # 
        self.model.load_state_dict(
            torch.load(
                os.path.join(self.pca_dir , "0.pt" ),
                map_location=torch.device(self.device),
            )
        )        

        return P, pca.explained_variance_ratio_

    def get_model_param_vec(self):
        """
        Return model parameters as a vector
        """
        vec = []
        for _,param in self.model.named_parameters():
            vec.append(param.detach().cpu().numpy().reshape(-1))
        return np.concatenate(vec, 0)

    def get_model_grad_vec(self):
        # Return the model grad as a vector

        vec = []
        for _,param in self.model.named_parameters():
            vec.append(param.grad.detach().reshape(-1))
        return torch.cat(vec, 0)

    def update_grad(self, grad_vec):
        idx = 0
        for _,param in self.model.named_parameters():
            arr_shape = param.grad.shape
            size = 1
            for i in range(len(list(arr_shape))):
                size *= arr_shape[i]
            param.grad.data = grad_vec[idx:idx+size].reshape(arr_shape)
            idx += size

    def update_param(self, param_vec):
        idx = 0
        for _,param in self.model.named_parameters():
            arr_shape = param.data.shape
            size = 1
            for i in range(len(list(arr_shape))):
                size *= arr_shape[i]
            param.data = param_vec[idx:idx+size].reshape(arr_shape)
            idx += size    


"""This implements a base class for clients."""


class BaseClient:
    """Abstract class of PPFL algorithm for client that trains local model.

    Args:
        id: unique ID for each client
        weight: aggregation weight assigned to each client
        model: (nn.Module): torch neural network model to train
        dataloader: PyTorch data loader
        device (str): device for computation
    """

    def __init__(
        self,
        id: int,
        weight: Dict,
        model: nn.Module,
        dataloader: DataLoader,
        cfg,
        outfile,
        test_dataloader,
    ):

        self.id = id
        self.weight = weight
        self.model = model
        self.dataloader = dataloader
        self.cfg = cfg
        self.outfile = outfile
        self.test_dataloader = test_dataloader

        self.primal_state = OrderedDict()
        self.dual_state = OrderedDict()
        self.primal_state_curr = OrderedDict()
        self.primal_state_prev = OrderedDict()

    def update(self):
        """Update local model parameters"""
        raise NotImplementedError

    def get_model(self):
        """Get the model

        Return:
            the ``state_dict`` of local model
        """
        return self.model.state_dict()

    def primal_residual_at_client(self, global_state) -> float:
        primal_res = 0
        for name, _ in self.model.named_parameters():
            primal_res += torch.sum(
                torch.square(global_state[name] - self.primal_state[name])
            )
        primal_res = torch.sqrt(primal_res).item()
        return primal_res

    def dual_residual_at_client(self) -> float:
        dual_res = 0
        if self.is_first_iter == 1:
            self.primal_state_curr = copy.deepcopy(self.primal_state)
            self.is_first_iter = 0

        else:
            self.primal_state_prev = copy.deepcopy(self.primal_state_curr)
            self.primal_state_curr = copy.deepcopy(self.primal_state)

            ## compute dual residual
            for name, _ in self.model.named_parameters():
                temp = self.penalty * (
                    self.primal_state_prev[name] - self.primal_state_curr[name]
                )

                dual_res += torch.sum(torch.square(temp))
            dual_res = torch.sqrt(dual_res).item()

        return dual_res

    def residual_balancing(self, prim_res, dual_res):

        if prim_res > self.residual_balancing.mu * dual_res:
            self.penalty = self.penalty * self.residual_balancing.tau
        if dual_res > self.residual_balancing.mu * prim_res:
            self.penalty = self.penalty / self.residual_balancing.tau

    def client_log_title(self):
        title = "%10s %10s %10s %10s %10s %10s %10s \n" % (
            "Round",            
            "LocalEpoch",
            "PerIter[s]",
            "TrainLoss",
            "TrainAccu",
            "TestLoss",
            "TestAccu",
        )
        self.outfile.write(title)
        self.outfile.flush()

    def client_log_content(
        self, t, per_iter_time, train_loss, train_accuracy, test_loss, test_accuracy
    ):
        contents = "%10s %10s %10.2f %10.4f %10.4f %10.4f %10.4f \n" % (
            self.round,
            t,
            per_iter_time,
            train_loss,
            train_accuracy,
            test_loss,
            test_accuracy,
        )
        self.outfile.write(contents)
        self.outfile.flush()


    def local_validation(self,  dataloader):
    
        self.model.eval()
        loss = 0
        correct = 0
        tmpcnt = 0
        tmptotal = 0     
        with torch.no_grad():
            for img, target in dataloader:
                tmpcnt += 1
                tmptotal += len(target)
                img = img.to(self.cfg.device)
                target = target.to(self.cfg.device)            
                output = self.model(img)      
                loss += eval(self.loss_fn)(output, target).item()                    
            
                pred = output.argmax(dim=1, keepdim=True)

                correct += pred.eq(target.view_as(pred)).sum().item()         
        
    

        loss = round( loss / tmpcnt, 4)
        accuracy = 100.0 * correct / tmptotal


        return loss, accuracy 

    def client_validation(self, dataloader):

        if dataloader is not None:
            self.loss_fn = eval(self.loss_type)
        else:
            self.loss_fn = None

        if self.loss_fn is None or dataloader is None:
            return 0.0, 0.0

        self.model.to(self.cfg.device)
        self.model.eval()
        loss = 0
        correct = 0
        tmpcnt = 0
        tmptotal = 0
        with torch.no_grad():
            for img, target in dataloader:
                tmpcnt += 1
                tmptotal += len(target)
                img = img.to(self.cfg.device)
                target = target.to(self.cfg.device)
                output = self.model(img)
                loss += self.loss_fn(output, target).item()

                if self.loss_type == "torch.nn.BCELoss()":
                    pred = torch.round(output)
                else:
                    pred = output.argmax(dim=1, keepdim=True)

                correct += pred.eq(target.view_as(pred)).sum().item()

        # FIXME: do we need to sent the model to cpu again?
        # self.model.to("cpu")

        loss = loss / tmpcnt
        accuracy = 100.0 * correct / tmptotal

        return loss, accuracy

    """ 
    Differential Privacy 
        (Laplacian mechanism) 
        - Noises from a Laplace dist. with zero mean and "scale_value" are added to the primal_state 
        - Variance = 2*(scale_value)^2
        - scale_value = sensitivty/epsilon, where sensitivity is determined by data, algorithm.
    """

    def laplace_mechanism_output_perturb(self, scale_value):
        """Differential privacy for output perturbation based on Laplacian distribution.
        This output perturbation adds Laplace noise to ``primal_state``.

        Args:
            scale_value: scaling vector to control the variance of Laplacian distribution
        """

        for name, param in self.model.named_parameters():
            mean = torch.zeros_like(param.data)
            scale = torch.zeros_like(param.data) + scale_value
            m = torch.distributions.laplace.Laplace(mean, scale)
            self.primal_state[name] += m.sample()


    """ 
    Projection Matrix
        Each client can obtain a trajectory of iterates by training a model using the local data, i.e., w^1, w^2, ..., w^I, where w^i in R^J
        By conduting PCA (principle component analysis) using the "I" iterates, one can obtain a projection matrix "M X J" where M ( < I ) is the number of components.
    """


    def log_pca(self):
        
        self.outfile.write("pca_dir: %s \n" %(self.pca_dir) )
        self.outfile.write("Ratio: %s \n" % (self.EVR))
        self.outfile.write("Sum: %s \n" % ( sum(self.EVR) ) )        
        self.outfile.write("P: (%s, %s) \n" % (self.P.shape[0], self.P.shape[1])) 
        self.outfile.flush() 

    def construct_projection_matrix(self):
  
        W = []
 
        for i in range(self.params_start, self.params_end):
            self.model.load_state_dict(
                torch.load(
                    os.path.join(self.pca_dir , "%s.pt" %(i) ),
                    map_location=torch.device(self.cfg.device),
                )
            )
            W.append(self.get_model_param_vec())

        W = np.array(W)
        
        # Obtain base variables through PCA
        pca = PCA(n_components=self.ncomponents)
        pca.fit_transform(W)
        P = np.array(pca.components_)
        P = torch.from_numpy(P).to(self.cfg.device)

        # 
        self.model.load_state_dict(
            torch.load(
                os.path.join(self.pca_dir , "0.pt" ),
                map_location=torch.device(self.cfg.device),
            )
        )

        return P, pca.explained_variance_ratio_

    def get_model_param_vec(self):
        """
        Return model parameters as a vector
        """
        vec = []
        for _,param in self.model.named_parameters():
            vec.append(param.detach().cpu().numpy().reshape(-1))
        return np.concatenate(vec, 0)

    def get_model_grad_vec(self):
        # Return the model grad as a vector

        vec = []
        for _,param in self.model.named_parameters():
            vec.append(param.grad.detach().reshape(-1))
        return torch.cat(vec, 0)

    def update_grad(self, grad_vec):
        idx = 0
        for _,param in self.model.named_parameters():
            arr_shape = param.grad.shape
            size = 1
            for i in range(len(list(arr_shape))):
                size *= arr_shape[i]
            param.grad.data = grad_vec[idx:idx+size].reshape(arr_shape)
            idx += size

    def update_param(self, param_vec):
        idx = 0
        for _,param in self.model.named_parameters():
            arr_shape = param.data.shape
            size = 1
            for i in range(len(list(arr_shape))):
                size *= arr_shape[i]
            param.data = param_vec[idx:idx+size].reshape(arr_shape)
            idx += size    
