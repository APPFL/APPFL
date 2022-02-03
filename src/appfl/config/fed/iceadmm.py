from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf

@dataclass
class ICEADMM:
    type: str = "iceadmm"
    servername: str = "ICEADMMServer"
    clientname: str = "ICEADMMClient"
    args: DictConfig = OmegaConf.create(
        {
            "num_local_epochs": 1,
            "accum_grad": True,
            "coeff_grad": True,

            ## Optimizer for a gradient calculation (esp., to use zero.grad)
            "optim": "SGD",
            "optim_args": {
                "lr": 0.01,
                # "momentum": 0.9,
                # "weight_decay": 1e-5,
            },

            # ADMM parameters  
            "init_penalty": 100.0,
            "init_proximity": 0,

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
