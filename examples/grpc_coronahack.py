import os
import time
import json
import torch

from appfl.config import *
from appfl.misc.data import *
from models.cnn import *
import appfl.run_grpc_server as grpc_server
import appfl.run_grpc_client as grpc_client
from mpi4py import MPI

DataSet_name = "Coronahack"
num_clients = 4
num_channel = 3  # 1 if gray, 3 if color
num_classes = 7  # number of the image classes
num_pixel = 32  # image size = (num_pixel, num_pixel)

dir = os.getcwd() + "/datasets/PreprocessedData/%s_Clients_%s" % (
    DataSet_name,
    num_clients,
)


def get_data(comm: MPI.Comm):
    # test data for a server
    with open("%s/all_test_data.json" % (dir)) as f:
        test_data_raw = json.load(f)

    test_dataset = Dataset(
        torch.FloatTensor(test_data_raw["x"]), torch.tensor(test_data_raw["y"])
    )

    # training data for multiple clients
    train_datasets = []

    for client in range(num_clients):
        with open("%s/all_train_data_client_%s.json" % (dir, client)) as f:
            train_data_raw = json.load(f)
        train_datasets.append(
            Dataset(
                torch.FloatTensor(train_data_raw["x"]),
                torch.tensor(train_data_raw["y"]),
            )
        )

    data_sanity_check(train_datasets, test_dataset, num_channel, num_pixel)
    return train_datasets, test_dataset


def get_model(comm: MPI.Comm):
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

    if comm_size > 1:
        # Try to launch both a server and clients.
        if comm_rank == 0:
            grpc_server.run_server(cfg, model, num_clients, test_dataset)
        else:
            grpc_client.run_client(cfg, comm_rank-1, model, train_datasets[comm_rank - 1], comm_rank)
        print("------DONE------", comm_rank)
    else:
        # Just launch a server.
        grpc_server.run_server(cfg, model, num_clients, test_dataset)


if __name__ == "__main__":
    main()
