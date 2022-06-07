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
from models.cnn import *
import appfl.run_funcx_server as funcx_server
import appfl.run_funcx_client as funcx_client
import argparse

from funcx import FuncXClient

"""
python grpc_mnist_client.py --host=localhost --client_id=0 --nclients=1
"""

""" read arguments """ 

parser = argparse.ArgumentParser() 

parser.add_argument('--device', type=str, default="cpu")    

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


def main():

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    """ Configuration """     
    cfg = OmegaConf.structured(Config)
    
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

    start_time = time.time()

    """ User-defined model """    
    model = CNN(args.num_channel, args.num_classes, args.num_pixel)
    loss_fn = torch.nn.CrossEntropyLoss()  

    cfg.validation = False
    ## loading models 
    cfg.load_model = False
    if cfg.load_model == True:
        cfg.load_model_dirname      = "./save_models"
        cfg.load_model_filename     = "Model"               
        model = load_model(cfg)      

    """ User-defined data """        
    train_datasets, test_dataset = get_data(comm)
    
    ## Sanity check for the user-defined data
    if cfg.data_sanity == True:
        data_sanity_check(train_datasets, test_dataset, args.num_channel, args.num_pixel)        

 
    print(
        "-------Loading_Time=",
        time.time() - start_time,
    ) 
    
    """ saving models """
    cfg.save_model = False
    if cfg.save_model == True:
        cfg.save_model_dirname      = "./save_models"
        cfg.save_model_filename     = "MNIST_CNN"    
 
    # Try to launch both a server and clients.
    funcx_server.run_server(cfg, model, args.num_clients)
    funcx_client.run_client(cfg, comm_rank-1, model, loss_fn, train_datasets[comm_rank - 1], comm_rank, test_dataset)
            
    print("------DONE------", comm_rank)

if __name__ == "__main__":
    # main()
    fxc = FuncXClient()
    tutorial_endpoint = '4b116d3c-1703-4f8f-9f6f-39921e5864df'
    funcx_client.run_client(fxc, tutorial_endpoint)
