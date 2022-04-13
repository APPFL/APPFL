import os
import time

import numpy as np
import torch

import torchvision
from torchvision.transforms import ToTensor

from appfl.config import *
from appfl.misc.data import *
from appfl.misc.utils import *
from models.cnn import *

import appfl.run_serial as rs
import appfl.run_mpi as rm

from mpi4py import MPI

import argparse

DataSet_name = "MNIST"
num_clients = 2
num_channel = 1  # 1 if gray, 3 if color
num_classes = 10  # number of the image classes
num_pixel = 28   # image size = (num_pixel, num_pixel)

dir = os.getcwd() + "/datasets/RawData"

def get_data(comm: MPI.Comm):
    comm_rank = comm.Get_rank()

    # Root download the data if not already available.
    if comm_rank == 0:
        # test data for a server
        test_data_raw = eval("torchvision.datasets." + DataSet_name)(
            dir, download=True, train=False, transform=ToTensor()
        )

    comm.Barrier()
    if comm_rank > 0:
        # test data for a server
        test_data_raw = eval("torchvision.datasets." + DataSet_name)(
            dir, download=False, train=False, transform=ToTensor()
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
    train_data_raw = eval("torchvision.datasets." + DataSet_name)(
        dir, download=False, train=True, transform=ToTensor()
    )

    split_train_data_raw = np.array_split(range(len(train_data_raw)), num_clients)
    train_datasets = []
    for i in range(num_clients):

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
    model = CNN(num_channel, num_classes, num_pixel)
    return model


## Run
def main():
    
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()


    parser = argparse.ArgumentParser()
    parser.add_argument('--server', type=str, required=True)    
    parser.add_argument('--num_epochs', type=int, required=True)    
    parser.add_argument('--client_lr', type=float, required=True)    

    parser.add_argument('--server_lr', type=float, required=False)    
    parser.add_argument('--mparam_1', type=float, required=False)    
    parser.add_argument('--mparam_2', type=float, required=False)    
    parser.add_argument('--adapt_param', type=float, required=False)    


    args = parser.parse_args()    

    """ Configuration """     
    cfg = OmegaConf.structured(Config) 
    cfg.device = "cuda"
    cfg.fed.servername = args.server
    cfg.num_epochs = args.num_epochs
    cfg.fed.args.optim_args.lr = args.client_lr

    cfg.output_filename += "_%s_%s_ClientLR_%s" %(DataSet_name, args.server, args.client_lr)
    
    if args.server_lr != None:
        cfg.fed.args.server_learning_rate = args.server_lr
        cfg.output_filename += "_ServerLR_%s" %(args.server_lr)
        
    if args.adapt_param != None:
        cfg.fed.args.server_adapt_param = args.adapt_param   
        cfg.output_filename += "_AdaptParam_%s" %(args.adapt_param)             
        
    if args.mparam_1 != None:
        cfg.fed.args.server_momentum_param_1 = args.mparam_1
        cfg.output_filename += "_MParam1_%s" %(args.mparam_1)
        
    if args.mparam_2 != None:
        cfg.fed.args.server_momentum_param_2 = args.mparam_2  
        cfg.output_filename += "_MParam2_%s" %(args.mparam_2)
        
        
    ## Reproducibility
    if cfg.reproduce == True:
        torch.manual_seed(1)
        torch.backends.cudnn.deterministic = True 

    start_time = time.time()

    """ User-defined model """    
    model = get_model(comm)
    cfg.fed.args.loss_type = "torch.nn.CrossEntropyLoss()"
    
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
        data_sanity_check(train_datasets, test_dataset, num_channel, num_pixel)        

    print(
        "--------Data and Model: Loading_Time=",
        time.time() - start_time,
    ) 
    
    """ saving models """
    cfg.save_model = False
    if cfg.save_model == True:
        cfg.save_model_dirname      = "./save_models"
        cfg.save_model_filename     = "Model"      
        
    
    cfg.summary_file = cfg.output_dirname + "/Summary_%s.txt" %(DataSet_name)
 
    
    """ Running """
    if comm_size > 1:
        if comm_rank == 0:
            rm.run_server(cfg, comm, model, num_clients, test_dataset, DataSet_name)
        else:
            rm.run_client(cfg, comm, model, num_clients, train_datasets)
        print("------DONE------", comm_rank)
    else:
        rs.run_serial(cfg, model, train_datasets, test_dataset, DataSet_name)
 
 
 

if __name__ == "__main__":
    main()


# To run CUDA-aware MPI:
# mpiexec -np 3 --mca opal_cuda_support 1 python ./mnist.py
# To run MPI:
# mpiexec -np 3 python ./mnist.py
# To run:
# python ./mnist.py


