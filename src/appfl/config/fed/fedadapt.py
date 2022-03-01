from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf

@dataclass
class FedAdapt:
    type: str = "fedadapt"  
    servername: str = "FedAdamServer" ## FedAvgServer, FedAvgMServer, FedAdagradServer, FedAdamServer, FedYogiServer
    clientname: str = "FedAdaptClient"
    args: DictConfig = OmegaConf.create(
        {
            "num_local_epochs": 10,

            ## Server update
            "server_learning_rate": 0.01,
            "server_adapt_param": 0.001,
            "server_momentum_param_1": 0.9,
            "server_momentum_param_2": 0.99,

            ## Optimizer
            "optim": "SGD",
            "optim_args": {
                "lr": 0.01,
                "momentum": 0.9,
                "weight_decay": 1e-5,
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
