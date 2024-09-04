from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf

@dataclass
class adaptive_fl:
    type: str = "adaptive_fl"
    servername: str = "AdaptiveFLServer"
    clientname: str = "ClientAdaptOptim"
    args: DictConfig = OmegaConf.create(
        {
            # add new arguments
            from .fed.adaptive_fl import *
        }
    )
    