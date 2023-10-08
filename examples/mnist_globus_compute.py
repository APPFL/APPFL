import torch
import argparse
from models.cnn  import *
from appfl.config import *
from appfl.misc.data import *
from appfl.misc.utils import *
from appfl.misc.logging import *
from globus_compute_sdk import Client
from appfl.run_globus_compute_server import run_server

"""
python mnist_gc.py --client_config path_to_client_config.yaml --server_config path_to_server_config.yaml
"""

""" read arguments """ 
parser = argparse.ArgumentParser()  
parser.add_argument("--client_config", type=str, default="globus_compute/configs_client/mnist.yaml")
parser.add_argument("--server_config", type=str, default= "globus_compute/configs_server/mnist_fedavg.yaml") 

## other agruments
parser.add_argument('--clients-test', action='store_true', default=False)
parser.add_argument('--reproduce', action='store_true', default=False) 
parser.add_argument('--load-model', action='store_true', default=False) 
parser.add_argument('--load-model-dirname', type=str, default= "")
parser.add_argument('--load-model-filename', type=str, default= "")
parser.add_argument('--use-tensorboard', action='store_true', default=False)
parser.add_argument('--save-model', action='store_true', default=False)
parser.add_argument('--save-model-state-dict', action='store_true', default=False)
parser.add_argument('--checkpoints-interval', type=float, default=2)

args = parser.parse_args()

def main():
    # Configuration    
    cfg = OmegaConf.structured(GlobusComputeConfig)
    cfg.reproduce = args.reproduce
    cfg.load_model = args.load_model
    cfg.load_model_dirname  = args.load_model_dirname
    cfg.load_model_filename = args.load_model_filename
    cfg.save_model_state_dict = args.save_model_state_dict
    cfg.save_model = args.save_model
    cfg.checkpoints_interval = args.checkpoints_interval
    cfg.use_tensorboard= args.use_tensorboard
    cfg.validation = True   
    if cfg.reproduce == True:
        set_seed(1)
    mode = 'clients_testing' if args.clients_test else 'train'

    # loading globus compute configs from file
    load_globus_compute_client_config(cfg, args.client_config)
    load_globus_compute_server_config(cfg, args.server_config)

    # config logger
    mLogging.config_logger(cfg)    
    
    """ Server-defined model """
    ModelClass = get_executable_func(cfg.get_model)()
    model = ModelClass(**cfg.model_kwargs) 
    loss_fn = get_loss_func(cfg)
    val_metric = get_executable_func(cfg.val_metric)

    if cfg.load_model == True:
        path = cfg.load_model_dirname + "/%s%s" % (cfg.load_model_filename, ".pt")
        model.load_state_dict(torch.load(path)) 
        model.eval()
    
    server_test_dataset = None
    server_val_dataset  = None
    gcc = Client()
    run_server(cfg, model, loss_fn, val_metric, gcc, server_test_dataset, server_val_dataset, mode=mode)

if __name__ == "__main__":
    main()