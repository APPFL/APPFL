from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf

@dataclass
class NewAlgo:
    type: str = "adaptive_fl"
    servername: str = "adaptive_fl_Server"
    clientname: str = "adaptive_fl_Client"
    args: DictConfig = OmegaConf.create(
        {
            # add new arguments
            from .fed.adaptive_fl import *
        }
    )
    