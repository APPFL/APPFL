import sys

sys.path.append("..")

import time

start_time = time.time()

## User-defined datasets
import numpy as np
import torch
import torchvision
from torchvision.transforms import ToTensor
from appfl.misc.data import *

DataSet_name = "MNIST" 
num_clients = 4
num_channel = 1    # 1 if gray, 3 if color
num_classes = 10   # number of the image classes 
num_pixel   = 28   # image size = (num_pixel, num_pixel)


# test data for a server
test_data_raw = eval("torchvision.datasets."+DataSet_name)(
    f"../datasets/RawData",
    download=True,
    train=False,
    transform=ToTensor()
) 

test_data_input = []
test_data_label = []
for idx in range(len(test_data_raw)):    
    test_data_input.append( test_data_raw[idx][0].tolist() )
    test_data_label.append( test_data_raw[idx][1] )

test_dataset = Dataset(
    torch.FloatTensor(test_data_input), torch.tensor(test_data_label)
)


# training data for multiple clients
train_data_raw = eval("torchvision.datasets."+DataSet_name)(
    f"../datasets/RawData",
    download=True,
    train=True,
    transform=ToTensor()
)

split_train_data_raw = np.array_split(range(len(train_data_raw)), num_clients)     
train_datasets = []
for i in range(num_clients):    

    train_data_input=[]; train_data_label=[]    
    for idx in split_train_data_raw[i]:        
        train_data_input.append(train_data_raw[idx][0].tolist())
        train_data_label.append(train_data_raw[idx][1])
    
    train_datasets.append(
        Dataset(
            torch.FloatTensor(train_data_input),
            torch.tensor(train_data_label),
        )
    )

data_sanity_check(train_datasets, test_dataset, num_channel, num_pixel)

## User-defined model
from examples.models.cnn import *

model = CNN(num_channel, num_classes, num_pixel)

print(
    "----------Loaded Datasets and Model----------Elapsed Time=",
    time.time() - start_time,
)

## Run
import appfl.run as rt
import hydra
from mpi4py import MPI
from omegaconf import DictConfig


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
        rt.run_serial(cfg, model, train_datasets, test_dataset, DataSet_name)


if __name__ == "__main__":
    main()


# To run CUDA-aware MPI:
# mpiexec -np 5 --mca opal_cuda_support 1 python ./mnist.py
# To run MPI:
# mpiexec -np 5 python ./mnist.py
# To run:
# python ./mnist.py
