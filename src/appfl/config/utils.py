from omegaconf import OmegaConf
from .config import *
import yaml
import os.path as osp
from ..misc.utils import *
def show():
    conf = OmegaConf.structured(Config)
    print(OmegaConf.to_yaml(conf))


def load_exct_func_cfg(cfg_dict):
    src = OmegaConf.create(
        ExecutableFunc(**cfg_dict))
    assert src.module != "" or  src.script_file  != "",  "Need to specify the executable function by (module, call) or script file"
    assert not (src.module != "" and src.script_file != ""), "Can only specify the executable function by (module, call) or script file but not both"
    assert src.call != "", "Need to specify the function's name by setting 'call: <func name>' in the config file"
    if src.script_file != "":
        with open(src.script_file) as fi:
            src.source = fi.read()
        assert len(src.source) > 0, "Source file is empty."
    return src

def load_funcx_config(cfg: FuncXConfig,
    config_file: str):
    assert osp.exists(config_file), "Config file {config_file} not found!"
    with open(config_file) as fi:
        data = yaml.load(fi, Loader = yaml.SafeLoader)

    ## Load module configs for get_model and get_dataset method
    if 'get_data' in data['func']:
        cfg.get_data = load_exct_func_cfg(data['func']['get_data'])
    
    if 'loss' in data:
        cfg.loss = data['loss']
    if 'train_data_batch_size' in data:
        cfg.train_data_batch_size = data['train_data_batch_size']
    if 'test_data_batch_size' in data:
        cfg.test_data_batch_size = data['test_data_batch_size']
    cfg.get_model= load_exct_func_cfg(data['func']['get_model'])
    
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
            client['get_data'] = load_exct_func_cfg(client['get_data'])
        if 'data_pipeline' in client:
            client['data_pipeline']= OmegaConf.create(client['data_pipeline'])
        # else:
        #     client['data_pipeline']= OmegaConf.create({})
        client_cfg = OmegaConf.structured(FuncXClientConfig(**client))
        cfg.clients.append(client_cfg)
    
    cfg.num_clients = len(cfg.clients)
    return cfg