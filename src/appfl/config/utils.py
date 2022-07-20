from omegaconf import OmegaConf
from .config import *
import yaml
import os.path as osp

def show():
    conf = OmegaConf.structured(Config)
    print(OmegaConf.to_yaml(conf))

def load_funcx_config(cfg: FuncXConfig,
    config_file: str):
    assert osp.exists(config_file), "Config file {config_file} not found!"
    with open(config_file) as fi:
        data = yaml.load(fi, Loader = yaml.SafeLoader)

    ## Load module configs for get_model and get_dataset method
    if 'get_data' in data['func']:
        cfg.get_data = OmegaConf.structured(ExecutableFunc(**data['func']['get_data']))
    cfg.get_model= OmegaConf.structured(ExecutableFunc(**data['func']['get_model']))
    
    ## Load FL algorithm configs
    cfg.fed =Federated()
    cfg.fed.servername = data['algorithm']['servername']
    cfg.fed.clientname = data['algorithm']['clientname']
    cfg.fed.args = OmegaConf.create(data['algorithm']['args'])
    ## Load training configs
    cfg.num_epochs         = data['training']['num_epochs']
    cfg.save_model_dirname = data['training']['save_model_dirname']
    cfg.save_model_filename= data['training']['save_model_filename']
    ## Load model configs
    cfg.model_kwargs       = data['model']
    ## Load dataset configs
    cfg.dataset  = data['dataset']['name']
    
def load_funcx_device_config(cfg: FuncXConfig, 
    config_file: str):
    assert osp.exists(config_file), "Config file {config_file} not found!"
    with open(config_file) as fi:
        data = yaml.load(fi, Loader = yaml.SafeLoader)
    
    ## Load configs for server
    cfg.server   = OmegaConf.structured(FuncXServerConfig(**data['server']))
    
    ## Load configs for clients
    for client in data["clients"]:
        if 'get_data' in client:
            client['get_data'] = OmegaConf.create(client['get_data'])
        client_cfg = OmegaConf.structured(FuncXClientConfig(**client))
        
        cfg.clients.append(client_cfg)
    
    cfg.num_clients = len(cfg.clients)
    return cfg