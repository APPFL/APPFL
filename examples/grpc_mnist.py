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
import appfl.run_grpc_server as grpc_server
import appfl.run_grpc_client as grpc_client
from mpi4py import MPI
import argparse

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
 
 
def get_data(comm: MPI.Comm):
    comm_rank = comm.Get_rank()

    if comm_rank == 0:
        # test data for a server
        test_data_raw = eval("torchvision.datasets." + args.dataset)(
            f"./datasets/RawData", download=True, train=False, transform=ToTensor()
        )

    comm.Barrier()
    if comm_rank > 0:
        # test data for a server
        test_data_raw = eval("torchvision.datasets." + args.dataset)(
            f"./datasets/RawData", download=False, train=False, transform=ToTensor()
        )

    test_data_input = []
    test_data_label = []
    for idx in range(len(test_data_raw)):
        test_data_input.append(test_data_raw[idx][0].tolist())
        test_data_label.append(test_data_raw[idx][1])

    test_dataset = Dataset(
        torch.FloatTensor(test_data_input), torch.tensor(test_data_label)
    )

    # training data for multiple clients
    train_data_raw = eval("torchvision.datasets." + args.dataset)(
        f"./datasets/RawData", download=False, train=True, transform=ToTensor()
    )

    split_train_data_raw = np.array_split(range(len(train_data_raw)), args.num_clients)
    train_datasets = []
    for i in range(args.num_clients):

        train_data_input = []
        train_data_label = []
        for idx in split_train_data_raw[i]:
            train_data_input.append(train_data_raw[idx][0].tolist())
            train_data_label.append(train_data_raw[idx][1])

        train_datasets.append(
            Dataset(
                torch.FloatTensor(train_data_input),
                torch.tensor(train_data_label),
            )
        )

    return train_datasets, test_dataset


def get_model(comm: MPI.Comm):
    ## User-defined model
    model = CNN(args.num_channel, args.num_classes, args.num_pixel)
    return model


def main():

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()


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
    model = get_model(comm)
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
 

    if comm_size > 1:
        # Try to launch both a server and clients.
        if comm_rank == 0:            
            grpc_server.run_server(cfg, model, loss_fn, args.num_clients)
        else:            
            print(comm_rank, len(train_datasets))
            grpc_client.run_client(cfg, comm_rank-1, model, loss_fn, train_datasets[comm_rank - 1], comm_rank, test_dataset)
            
        print("------DONE------", comm_rank)
    else:
        # Just launch a server.
        grpc_server.run_server(cfg, model, args.num_clients, test_dataset)


if __name__ == "__main__":
    main()
