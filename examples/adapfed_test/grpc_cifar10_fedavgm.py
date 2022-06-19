import time

import numpy as np
import torch
import torchvision
from torchvision.transforms import ToTensor

from appfl.config import *
from appfl.misc.data import *
from models.cnn import *
import appfl.run_grpc_server as grpc_server
import appfl.run_grpc_client as grpc_client
from mpi4py import MPI

DataSet_name = "CIFAR10"
num_clients = 4
num_channel = 3    # 1 if gray, 3 if color
num_classes = 10   # number of the image classes
num_pixel   = 32   # image size = (num_pixel, num_pixel)

def get_data(comm : MPI.COMM_WORLD):
    comm_rank = comm.Get_rank()

    # Root download the data if not already available.
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

def main():
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

    # read default configuration
    cfg = OmegaConf.structured(Config)

    # Configuration for program
    cfg.device = "cuda"
    cfg.num_epochs = 1000
    cfg.output_dirname = "output/fed_avgm"

    # Configuration for FL
    cfg.fed.servername = "ServerFedAvgMomentum"
    cfg.fed.args.num_local_epochs = 1

    loss_fn = torch.nn.CrossEntropyLoss()


    if comm_size > 1:
        # Try to launch both a server and clients.
        if comm_rank == 0:
            grpc_server.run_server(cfg, model, loss_fn, num_clients, test_dataset)
        else:
            grpc_client.run_client(cfg, comm_rank-1, model, loss_fn, train_datasets[comm_rank - 1], comm_rank)
        print("------DONE------", comm_rank)
    else:
        # Just launch a server.
        grpc_server.run_server(cfg, model, loss_fn, num_clients, test_dataset)

if __name__ == "__main__":
    main()
