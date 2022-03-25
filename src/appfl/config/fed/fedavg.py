from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf


@dataclass
class FedAvg:
    type: str = "fedavg"
    servername: str = "FedAvgServer"
    clientname: str = "FedAvgClient"
    args: DictConfig = OmegaConf.create(
        {
            # User-defined loss function (pytorch format)
            "loss_type": "torch.nn.CrossEntropyLoss()",  ##  "torch.nn.CrossEntropyLoss()"  "torch.nn.BCELoss()"
            ## Clients optimizer
            "optim": "SGD",
            "num_local_epochs": 1,
            "optim_args": {
                "lr": 0.01,
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
