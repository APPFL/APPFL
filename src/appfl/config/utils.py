from omegaconf import OmegaConf
from .config import *
import yaml

def show():
    conf = OmegaConf.structured(Config)
    print(OmegaConf.to_yaml(conf))

def load_funcx_config(cfg: FuncXConfig, 
    config_file: str):
    with open(config_file) as fi:
        data = yaml.load(fi, Loader = yaml.SafeLoader)
    for client in data["clients"]:
        client_cfg = OmegaConf.structured(FuncXClientConfig(**client))
        cfg.clients.append(client_cfg)
    return cfg