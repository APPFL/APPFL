import sys
import os
import time

## User-defined datasets
import json
import numpy as np
import torch

from appfl.misc.data import *
from models.cnn import *
import appfl.run as rt
import hydra
from mpi4py import MPI
from omegaconf import DictConfig

DataSet_name = "FEMNIST"
num_clients = 203
num_channel = 1  # 1 if gray, 3 if color
num_classes = 62  # number of the image classes
num_pixel = 28  # image size = (num_pixel, num_pixel)

dir = os.getcwd() + "/datasets/RawData/%s" % (DataSet_name)

def get_data(comm : MPI.COMM_WORLD):
    # test data for a server
    test_data_raw = {}
    test_data_input = []
    test_data_label = []
    for idx in range(36):
        with open("%s/test/all_data_%s_niid_05_keep_0_test_9.json" % (dir, idx)) as f:
            test_data_raw[idx] = json.load(f)

        for client in test_data_raw[idx]["users"]:

            for data_input in test_data_raw[idx]["user_data"][client]["x"]:
                data_input = np.asarray(data_input)
                data_input.resize(28, 28)
                test_data_input.append([data_input])

            for data_label in test_data_raw[idx]["user_data"][client]["y"]:
                test_data_label.append(data_label)

    test_dataset = Dataset(
        torch.FloatTensor(test_data_input), torch.tensor(test_data_label)
    )

    # training data for multiple clients
    train_data_raw = {}
    train_datasets = []
    for idx in range(36):
        with open("%s/train/all_data_%s_niid_05_keep_0_train_9.json" % (dir, idx)) as f:
            train_data_raw[idx] = json.load(f)

        for client in train_data_raw[idx]["users"]:

            train_data_input_resize = []
            for data_input in train_data_raw[idx]["user_data"][client]["x"]:
                data_input = np.asarray(data_input)
                data_input.resize(28, 28)
                train_data_input_resize.append([data_input])

            train_datasets.append(
                Dataset(
                    torch.FloatTensor(train_data_input_resize),
                    torch.tensor(train_data_raw[idx]["user_data"][client]["y"]),
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

    ## Reproducibility
    torch.manual_seed(1)    
    torch.backends.cudnn.deterministic=True

    start_time = time.time()
    train_datasets, test_dataset = get_data(comm)
    model = get_model(comm)
    print(
        "----------Loaded Datasets and Model----------Elapsed Time=",
        time.time() - start_time,
    )

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
# mpiexec -np 5 --mca opal_cuda_support 1 python ./femnist.py
# To run MPI:
# mpiexec -np 5 python ./femnist.py
# To run:
# python ./femnist.py
