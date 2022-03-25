from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf
import torch

""" various server updates are available 
    ServerFedAvg
    ServerFedAvgMomentum
    ServerFedAdagrad
    ServerFedAdam 
    ServerFedYogi
"""


@dataclass
class Federated:
    type: str = "federated"
    servername: str = "ServerFedAvg"
    clientname: str = "ClientOptim"
    args: DictConfig = OmegaConf.create(
        {
            # User-defined loss function (pytorch format)
            "loss_type": "torch.nn.BCELoss()",
            ## Server update
            "server_learning_rate": 0.01,
            "server_adapt_param": 0.001,
            "server_momentum_param_1": 0.9,
            "server_momentum_param_2": 0.99,
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
