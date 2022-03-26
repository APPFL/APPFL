from omegaconf import OmegaConf
from .config import *


def show():
    conf = OmegaConf.structured(Config)
    print(OmegaConf.to_yaml(conf))
