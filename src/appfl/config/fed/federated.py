from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf
import torch

""" Numerous global update techniques are implemented:        
    (1) ServerFedAvg            : averaging local model parameters to update global model parameters    
    (2) ServerFedAvgMomentum    : ServerFedAvg with a momentum        
    (3) ServerFedAdagrad        : use of the adaptive gradient (Adagrad) algorithm for a global update at a server
    (4) ServerFedAdam           : use of the adaptive moment estimation (Adam) algorithm for a global update 
    (5) ServerFedYogi           : use of the Yogi algorithm for a global update                        
    (6) ServerFedSDLBFGS        : use of the stochastic dampened L-BFGS for a global update at a server
        
    At server, the global update is done by
            global_model_parameter += (server_learning_rate * m) / ( sqrt(v) + server_adapt_param)
    where
            server_learning_rate: learning rate for the global update 
            server_adapt_param: adaptivity parameter {1e-3, 1e-4, 1e-5}            
            m: momentum 
                m <- server_momentum_param_1 * m + (1-server_momentum_param_1) * PseudoGrad
                PseudoGrad <- pseudo gradient obtained by averaging differences of global and local model parameters
            v: variance
                For ``ServerFedAdagrad``,       v <- v + (PseudoGrad)^2  
                For ``ServerFedAdam``,          v <- server_momentum_param_2 * v + (1-server_momentum_param_2)*PseudoGrad^2
                For ``ServerFedYogi``,          v <- v - (1-server_momentum_param_2)*PseudoGrad^2 sign(v - PseudoGrad^2)
              
    See the following paper for more details on the above global update techniques:
    ``Reddi, S., Charles, Z., Zaheer, M., Garrett, Z., Rush, K., Konečný, J., Kumar, S. and McMahan, H.B., 2020. Adaptive federated optimization. arXiv preprint arXiv:2003.00295``
"""


@dataclass
class Federated:
    type: str = "federated"
    servername: str = "ServerFedAvg"
    clientname: str = "ClientOptim"
    args: DictConfig = OmegaConf.create(
        {
            ## Server update
            "server_learning_rate": 0.01,
            "server_adapt_param": 0.001,
            "server_momentum_param_1": 0.9,
            "server_momentum_param_2": 0.99,
            ## SD L-BFGS specific configurations
            "history": 10,
            "delta": 0.1,
            "max_step_size": 100,
            "increment": 0.9, 
            "search_control": 0.5,
            ## Clients optimizer
            "optim": "SGD",
            "num_local_epochs": 10,
            "optim_args": {
                "lr": 0.001,
            },
            ## Differential Privacy
            ##  epsilon: False  (non-private)
            ##  epsilon: 1      (stronger privacy as the value decreases)
            ##  epsilon: 0.05
            "epsilon": False,
            ## Gradient Clipping
            ## clip_value: False (no-clipping)
            ## clip_value: 10    (clipping)
            ## clip_value: 1
            "clip_value": False,
            "clip_norm": 1,
        }
    )
