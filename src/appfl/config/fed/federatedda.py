from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf

@dataclass
class FDA:
    type: str = "FDA"
    servername: str = "FDAServer"
    clientname: str = "FDAClient"
    args: DictConfig = OmegaConf.create(
        {
            # add new arguments
            # TO-DO
        }
    )