import sys

sys.path.insert(0, "..")

import time

## User-defined datasets
import numpy as np
import torch
import torchvision
from torchvision.transforms import ToTensor

from src.appfl.misc.data import *
from examples.models.cnn import *
import src.appfl.run as rt
import src.appfl.run_grpc_server as grpc_server
import src.appfl.run_grpc_client as grpc_client
import hydra
from mpi4py import MPI
from omegaconf import DictConfig

DataSet_name = "MNIST"
num_clients = 4
num_channel = 1    # 1 if gray, 3 if color
num_classes = 10   # number of the image classes
num_pixel   = 28   # image size = (num_pixel, num_pixel)

def get_data(comm : MPI.COMM_WORLD):
    comm_rank = comm.Get_rank()

    if comm_rank == 0:
        # test data for a server
        test_data_raw = eval("torchvision.datasets."+DataSet_name)(
            f"./datasets/RawData",
            download=True,
            train=False,
            transform=ToTensor()
        )

    comm.Barrier()
    if comm_rank > 0:
        # test data for a server
        test_data_raw = eval("torchvision.datasets."+DataSet_name)(
            f"./datasets/RawData",
            download=False,
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
        f"./datasets/RawData",
        download=False,
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
    return train_datasets, test_dataset

def get_model(comm : MPI.COMM_WORLD):
    ## User-defined model
    model = CNN(num_channel, num_classes, num_pixel)
    return model

## Run
@hydra.main(config_path="../src/appfl/config", config_name="config")
def main(cfg: DictConfig):

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    torch.manual_seed(1)

    start_time = time.time()
    train_datasets, test_dataset = get_data(comm)
    model = get_model(comm)
    print(
        "----------Loaded Datasets and Model----------Elapsed Time=",
        time.time() - start_time,
    )

    if comm_size > 1:
        # Try to launch both a server and clients.
        if comm_rank == 0:
            grpc_server.run_server(cfg, comm_rank, model, test_dataset, num_clients, DataSet_name)
        else:
            grpc_client.run_client(cfg, comm_rank, model, train_datasets[comm_rank-1])
        print("------DONE------", comm_rank)
    else:
        # Just launch a server.
        grpc_server.run_server(cfg, comm_rank, model, test_dataset, num_clients, DataSet_name)

if __name__ == "__main__":
    main()


# To run CUDA-aware MPI:
# mpiexec -np 5 --mca opal_cuda_support 1 python ./mnist.py
# To run MPI:
# mpiexec -np 5 python ./mnist.py
# To run:
# python ./mnist.py
