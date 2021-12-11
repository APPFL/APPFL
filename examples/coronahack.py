
import sys

sys.path.append("..")

import time

start_time = time.time()

## User-defined datasets
import json
import numpy as np
import torch
from appfl.misc.data import *


DataSet_name = "Coronahack"
num_clients = 4
num_channel = 3    # 1 if gray, 3 if color
num_classes = 7   # number of the image classes
num_pixel = 32   # image size = (num_pixel, num_pixel)

dir = "../datasets/ProcessedData/%s_Clients_%s" % (DataSet_name, num_clients)

# test data for a server
with open("%s/all_test_data.json"%(dir)) as f:    
    test_data_raw = json.load(f)    

test_dataset = Dataset(
    torch.FloatTensor(test_data_raw["x"]), torch.tensor(test_data_raw["y"])
)


# training data for multiple clients
train_datasets = []

for client in range(num_clients):    
    with open("%s/all_train_data_client_%s.json"%(dir,client)) as f:    
        train_data_raw = json.load(f)    
    train_datasets.append( Dataset( torch.FloatTensor(train_data_raw["x"]), torch.tensor(train_data_raw["y"]) ) )
    
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
# mpiexec -np 5 --mca opal_cuda_support 1 python ./coronahack.py
# To run MPI:
# mpiexec -np 5 python ./coronahack.py
# To run:
# python ./coronahack.py
