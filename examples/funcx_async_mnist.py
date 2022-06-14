import time
import os.path as osp

import numpy as np
import torch
import torchvision
from torchvision.transforms import ToTensor
from appfl.config import *
from appfl.misc.data import *
from appfl.misc.utils import *
from models.cnn  import *
import appfl.run_funcx_async_server as funcx_async_server
import argparse

from funcx import FuncXClient

"""
python grpc_mnist_client.py --host=localhost --client_id=0 --nclients=1
"""

""" read arguments """ 

parser = argparse.ArgumentParser()  
## appfl-funcx
parser.add_argument("--config", type=str, default="configs/funcx_mnist.yaml")

## dataset 
parser.add_argument('--num_channel', type=int, default=1)   
parser.add_argument('--num_classes', type=int, default=10)   
parser.add_argument('--num_pixel', type=int, default=28)   

## clients
parser.add_argument('--client_optimizer', type=str, default="Adam")    
parser.add_argument('--client_lr', type=float, default=1e-3)    
parser.add_argument('--num_local_epochs', type=int, default=3)    

## server
parser.add_argument('--server', type=str, default="ServerFedAvg")    
parser.add_argument('--num_epochs', type=int, default=2)    

parser.add_argument('--server_lr', type=float, required=False)    
parser.add_argument('--mparam_1', type=float, required=False)    
parser.add_argument('--mparam_2', type=float, required=False)    
parser.add_argument('--adapt_param', type=float, required=False)   

## tensorboard
parser.add_argument('--use_tensorboard', type=bool, default=True)
 
args = parser.parse_args()

def main():
    """ Configuration """     
    cfg = OmegaConf.structured(FuncXConfig)
 
    cfg.reproduce = True
    if cfg.reproduce == True:
        set_seed(1)

    ## loading funcX configs from file
    load_funcx_config(cfg, args.config)
    
    ## clients
    cfg.fed.args.optim = args.client_optimizer
    cfg.fed.args.optim_args.lr = args.client_lr
    cfg.fed.args.num_local_epochs = args.num_local_epochs
    
    ## server
    cfg.fed.servername = args.server
    cfg.num_epochs     = args.num_epochs

    ## tensorboard
    cfg.use_tensorboard= args.use_tensorboard
    
    ## outputs
    cfg.output_dirname = osp.join(cfg.server.output_dir, "outputs_%s_%s_%s"%(cfg.dataset, args.server, args.client_optimizer)) 

    ## validation
    cfg.validation = True

    ## loading models 
    cfg.load_model = False
    if cfg.load_model == True:
        cfg.load_model_dirname      = "./save_models"
        cfg.load_model_filename     = "Model"               
        model = load_model(cfg)      
    
    """ User-defined model """   
    cfg.model_args = [args.num_channel, args.num_classes, args.num_pixel] 
    ModelClass     = get_executable_func(cfg.get_model)()
    model          = ModelClass(*cfg.model_args, **cfg.model_kwargs) 
    loss_fn        = torch.nn.CrossEntropyLoss()  

    """ User-defined data """
    # Prepare dataset for testing at server 
    test_data_input = []
    test_data_label = []

    test_data_raw = eval("torchvision.datasets." + cfg.dataset)(
            osp.join(cfg.server.data_dir, "RawData"), download=True, train=False, transform=ToTensor()
        )

    for idx in range(len(test_data_raw)):
        test_data_input.append(test_data_raw[idx][0].tolist())
        test_data_label.append(test_data_raw[idx][1])

    test_dataset = Dataset(
        torch.FloatTensor(test_data_input), torch.tensor(test_data_label)
    )

    """ Saving models """
    cfg.save_model = False
    if cfg.save_model == True:
        cfg.save_model_dirname      = "./save_models"
        cfg.save_model_filename     = "MNIST_CNN"   
    
    """ APPFL with funcX """
    ## create funcX client object
    fxc = FuncXClient(batch_enabled=True)
    ## run funcX server
    funcx_async_server.run_server(cfg, model, loss_fn, fxc , test_dataset)

if __name__ == "__main__":
    main()