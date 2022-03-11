import os
import time

import numpy as np
import torch

import torchvision
from torchvision.transforms import ToTensor

from appfl.config import *
from appfl.misc.data import *
from models.cnn import *

import appfl.run as rt
from mpi4py import MPI

import argparse

DataSet_name = "MNIST"
num_clients = 2
num_channel = 1  # 1 if gray, 3 if color
num_classes = 10  # number of the image classes
num_pixel = 28  # image size = (num_pixel, num_pixel)

dir = os.getcwd() + "/datasets/RawData"

def get_data(comm: MPI.Comm):
    comm_rank = comm.Get_rank()

    if comm_rank == 0:        
        test_dataset = eval("torchvision.datasets." + DataSet_name)(
            dir, download=True, train=False, transform=ToTensor()
        )

    comm.Barrier()
    if comm_rank > 0:        
        test_dataset = eval("torchvision.datasets." + DataSet_name)(
            dir, download=False, train=False, transform=ToTensor()
        ) 

    # training data for multiple clients
    train_data_raw = eval("torchvision.datasets." + DataSet_name)(
        dir, download=False, train=True, transform=ToTensor()
    )

    split_train_data_raw = np.array_split(range(len(train_data_raw)), num_clients)
    train_datasets = []
    for i in range(num_clients):
        train_datasets.append(torch.utils.data.Subset(train_data_raw, split_train_data_raw[i]))
        
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

    """ Configuration """     
    cfg = OmegaConf.structured(Config) 

    parser = argparse.ArgumentParser()
    parser.add_argument('--server', type=str, required=False)    
    args = parser.parse_args()    

    cfg.fed.servername = args.server

    ## Reproducibility
    if cfg.reproduce == True:
        torch.manual_seed(1)
        torch.backends.cudnn.deterministic = True 

    """ User-defined model and data """
    start_time = time.time()
 
    model = get_model(comm)
    cfg.fed.args.loss_type = "torch.nn.CrossEntropyLoss()"
 
    train_datasets, test_dataset = get_data(comm)

    ## Sanity check for the user-defined data
    if cfg.data_sanity == True:
        data_sanity_check(train_datasets, test_dataset, num_channel, num_pixel)        

    print(
        "--------Data and Model: Loading_Time=",
        time.time() - start_time,
    )
 
    
    """ Running """
    if comm_size > 1:
        if comm_rank == 0:
            rt.run_server(cfg, comm, model, num_clients, test_dataset, DataSet_name)
        else:
            rt.run_client(cfg, comm, model, num_clients, train_datasets)
        print("------DONE------", comm_rank)
    else:
        rt.run_serial(cfg, model, train_datasets, test_dataset, DataSet_name)


if __name__ == "__main__":
    main()


# To run CUDA-aware MPI:
# mpiexec -np 5 --mca opal_cuda_support 1 python ./mnist.py
# To run MPI:
# mpiexec -np 5 python ./mnist.py
# To run:
# python ./mnist.py
