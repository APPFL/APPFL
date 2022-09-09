import os
import time
import json
import torch

from appfl.config import *
from appfl.misc.data import *
from models.utils import get_model
import appfl.run_serial as rs
import appfl.run_mpi as rm
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


def main():
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    ## Reproducibility
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()
    train_datasets, test_dataset = get_data(comm)

    args = {}
    args.num_channel = num_channel
    args.num_classes = num_classes
    args.num_pixel = num_piexl
    model = get_model(args)
    loss_fn = torch.nn.CrossEntropyLoss()   

    print(
        "----------Loaded Datasets and Model----------Elapsed Time=",
        time.time() - start_time,
    )

    # read default configuration
    cfg = OmegaConf.structured(Config)

    if comm_size > 1:
        if comm_rank == 0:
            rm.run_server(cfg, comm, loss_fn, model, num_clients, test_dataset, DataSet_name)
        else:
            rm.run_client(cfg, comm, loss_fn, model, num_clients, train_datasets)
        print("------DONE------", comm_rank)
    else:
        rs.run_serial(cfg, model, loss_fn, train_datasets, test_dataset, DataSet_name)


if __name__ == "__main__":
    main()


# To run CUDA-aware MPI:
# mpiexec -np 5 --mca opal_cuda_support 1 python ./coronahack.py
# To run MPI:
# mpiexec -np 5 python ./coronahack.py
# To run:
# python ./coronahack.py
