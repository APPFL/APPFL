from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf

@dataclass
class FedAvg:
    type: str = "fedavg"
    servername: str = "FedAvgServer"
    clientname: str = "FedAvgClient"
    args: DictConfig = OmegaConf.create(
        {
            "num_local_epochs": 1,

            # User-defined loss function (pytorch format)
            "loss_type": "torch.nn.CrossEntropyLoss()", ##  "torch.nn.CrossEntropyLoss()"  "torch.nn.BCELoss()"

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
