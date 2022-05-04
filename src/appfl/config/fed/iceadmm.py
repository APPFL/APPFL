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
            ## Penalty
            "init_penalty": 100.0,
            ## Adaptive penalty
            ## Residual balancing (see (3.13) in "Distributed Optimization and Statistical Learning via the ADMM")
            ## penalty = penalty*(tau) if prim_res > (mu)*dual_res
            ## penalty = penalty/(tau) if dual_res > (mu)*prim_res
            "residual_balancing": {
                "res_on": True,
                "res_on_every_update": True,
                "tau": 2,
                "mu": 2,
            },
            ## Proximal term
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
