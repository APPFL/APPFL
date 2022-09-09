import time
import os.path as osp

import numpy as np
import torch
import torchvision
from torchvision.transforms import ToTensor
from appfl.config import *
from appfl.misc.data import *
from appfl.misc.utils import *
from appfl.misc.logging import *
from models.cnn  import *
import appfl.run_funcx_server as funcx_server
import argparse

from funcx import FuncXClient

"""
python grpc_mnist_client.py --host=localhost --client_id=0 --nclients=1
"""

""" read arguments """ 

parser = argparse.ArgumentParser()  
## appfl-funcx
parser.add_argument("--client_config", type=str, required= True)
parser.add_argument("--config", type=str, default= "configs/fed_async/funcx_fed_async_mnist.yaml") 

## other agruments
parser.add_argument('--reproduce', action='store_true', default=True) 
parser.add_argument('--use_tensorboard', action='store_true', default=True)
 
args = parser.parse_args()

def main():
    """ Configuration """     
    cfg = OmegaConf.structured(FuncXConfig)
 
    cfg.reproduce = True
    cfg.save_model_state_dict = True
    cfg.save_model = True
    cfg.checkpoints_interval = 2
    if cfg.reproduce == True:
        set_seed(1)

    ## loading funcX configs from file
    load_funcx_device_config(cfg, args.client_config)
    load_funcx_config(cfg, args.config)

    ## using funcx ClientOptimizer object
    cfg.fed.clientname = "FuncxClientOptim"
    ## tensorboard
    cfg.use_tensorboard= args.use_tensorboard
    
    ## config logger
    mLogging.config_logger(cfg,
        osp.basename(args.config), osp.basename(args.client_config)
    )

    ## validation
    cfg.validation = True   
    
    """ User-defined model """
    ModelClass     = get_executable_func(cfg.get_model)()
    model          = ModelClass(*cfg.model_args, **cfg.model_kwargs) 
    loss_fn        = get_loss_func(cfg.loss)

    """ User-defined data """
    ## save a copy of config to logfile
    logger = mLogging.get_logger()
    logger.info(
        OmegaConf.to_yaml(cfg)
    )
    
    """ Prepare test dataset"""
    server_test_dataset = None
    server_val_dataset  = None
    """ APPFL with funcX """
    ## create funcX client object
    fxc = FuncXClient()
    ## run funcX server
    funcx_server.run_server(cfg, model, loss_fn, fxc, server_test_dataset, server_val_dataset)

if __name__ == "__main__":
    main()