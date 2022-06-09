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
    ## Load module configs for get_model and get_dataset method
    cfg.get_data = OmegaConf.structured(ExecutableFunc(**data['func']['get_data']))
    cfg.get_model= OmegaConf.structured(ExecutableFunc(**data['func']['get_model']))

    ## Load configs for server
    cfg.server   = OmegaConf.structured(FuncXServerConfig(**data['server']))
    cfg.dataset  = data['dataset']['name']
    
    ## Load configs for clients
    for client in data["clients"]:
        client_cfg = OmegaConf.structured(FuncXClientConfig(**client))
        cfg.clients.append(client_cfg)
    
    cfg.num_clients = len(cfg.clients)
    return cfg