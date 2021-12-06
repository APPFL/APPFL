import sys

sys.path.append("..")
import time
start_time = time.time()
## User-defined datasets
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from examples.ReadDataset import *

DataSet_name = "MNIST" 
num_clients = 4
num_channel = 1    # 1 if gray, 3 if color
num_classes = 10   # number of the image classes 
num_pixel   = 28   # image size = (num_pixel, num_pixel)

train_datasets, test_dataset = ReadDataset(DataSet_name, num_clients, num_channel, num_pixel)


## User-defined model
from examples.cnn import *
 
model = CNN(num_channel, num_classes, num_pixel)

print("----------Loaded Datasets and Model----------Elapsed Time=",time.time()-start_time )

## train
import appfl.run as rt
import hydra
from omegaconf import DictConfig
from mpi4py import MPI

@hydra.main(config_path="../appfl/config", config_name="config")
def main(cfg: DictConfig):
    
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
 
    torch.manual_seed(1)
 
    if comm_size > 1:
        if comm_rank == 0:            
            rt.run_server(cfg, comm, model, test_dataset, num_clients, DataSet_name)
        else:                        
            rt.run_client(cfg, comm, model, train_datasets, num_clients)
        print("------DONE------", comm_rank)
    else:
        rt.run_serial(cfg, model, train_datasets, test_dataset)

if __name__ == "__main__":
    main()


# To run CUDA-aware MPI:
# mpiexec -np 5 --mca opal_cuda_support 1 python ./mnist.py
# To run MPI:
# mpiexec -np 5 python ./mnist.py
# To run:
# python ./mnist.py
