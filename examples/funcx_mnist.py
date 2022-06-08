import sys
import time
import logging
 
import numpy as np
import torch
import torchvision
from torchvision.transforms import ToTensor

from appfl.config import *
from appfl.misc.data import *
from appfl.misc.utils import *
from appfl.funcx import get_model
from models.cnn  import *
import appfl.run_funcx_server as funcx_server
import argparse

from funcx import FuncXClient

"""
python grpc_mnist_client.py --host=localhost --client_id=0 --nclients=1
"""

""" read arguments """ 

parser = argparse.ArgumentParser() 

parser.add_argument('--device', type=str, default="cpu")    
## appfl-funcx
parser.add_argument("--config", type=str, default="configs/funcx_mnist.yaml")

## dataset
parser.add_argument('--dataset', type=str, default="MNIST")   
parser.add_argument('--num_channel', type=int, default=1)   
parser.add_argument('--num_classes', type=int, default=10)   
parser.add_argument('--num_pixel', type=int, default=28)   

## clients
parser.add_argument('--num_clients', type=int, default=1)    
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
 
args = parser.parse_args()    

if torch.cuda.is_available():
    args.device="cuda"

def check_endpoint(fxc, endpoints):
    for endpoint in endpoints:
        print("------ Status of Endpoint %s ------" % endpoint)
        endpoint_status = fxc.get_endpoint_status(endpoint)
        print("Status       : %s" % endpoint_status['status'])
        print("Workers      : %s" % endpoint_status['logs'][0]['info']['total_workers'])
        print("Pending tasks: %s" % endpoint_status['logs'][0]['info']['pending_tasks'])
def main():
    logging.basicConfig(stream=sys.stdout, level=None) #logging.INFO
    """ Configuration """     
    cfg = OmegaConf.structured(FuncXConfig)
    
    cfg.device = args.device 
    cfg.reproduce = True
    if cfg.reproduce == True:
        set_seed(1)

    ## clients
    cfg.num_clients = args.num_clients
    cfg.fed.args.optim = args.client_optimizer
    cfg.fed.args.optim_args.lr = args.client_lr
    cfg.fed.args.num_local_epochs = args.num_local_epochs
    
    ## server
    cfg.fed.servername = args.server
    cfg.num_epochs = args.num_epochs

    ## outputs
    cfg.output_dirname = "./outputs_%s_%s_%s"%(args.dataset, args.server, args.client_optimizer)     
    cfg.output_filename = "result"    

    ## validation
    cfg.validation = False

    ## loading models 
    cfg.load_model = False
    if cfg.load_model == True:
        cfg.load_model_dirname      = "./save_models"
        cfg.load_model_filename     = "Model"               
        model = load_model(cfg)      

    ## loading funcX configs from file
    load_funcx_config(cfg, args.config)
    
    """ User-defined model """   
    cfg.model_args = [args.num_channel, args.num_classes, args.num_pixel] 
    ModelClass     = get_model()
    model          = ModelClass(*cfg.model_args, **cfg.model_kwargs) 
    loss_fn        = torch.nn.CrossEntropyLoss()  

    """ User-defined data """        
    cfg.dataset = args.dataset

    """ Saving models """
    cfg.save_model = False
    if cfg.save_model == True:
        cfg.save_model_dirname      = "./save_models"
        cfg.save_model_filename     = "MNIST_CNN"   
    
    """ APPFL with funcX """
    ## create funcX client object
    fxc = FuncXClient()
    
    ## run server
    funcx_server.run_server(cfg, model, loss_fn, fxc)
    
    # check_endpoint(fxc, endpoints)
    # train_datasets, test_dataset = get_data(comm)
    
    # ## Sanity check for the user-defined data
    # if cfg.data_sanity == True:
    #     data_sanity_check(train_datasets, test_dataset, args.num_channel, args.num_pixel)        

 
    # print(
    #     "-------Loading_Time=",
    #     time.time() - start_time,
    # )  
 
    # # Try to launch both a server and clients.
    # funcx_server.run_server(cfg, model, args.num_clients)
    # funcx_client.run_client(cfg, comm_rank-1, model, loss_fn, train_datasets[comm_rank - 1], comm_rank, test_dataset)
            
    # print("------DONE------", comm_rank)

if __name__ == "__main__":
    main()