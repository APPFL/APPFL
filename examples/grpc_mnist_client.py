import sys
import time
import logging
import argparse

## User-defined datasets
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

DataSet_name = "MNIST"
num_channel = 1  # 1 if gray, 3 if color
num_classes = 10  # number of the image classes
num_pixel = 28  # image size = (num_pixel, num_pixel)


def get_data(num_clients: int):

    # training data for multiple clients
    train_data_raw = eval("torchvision.datasets." + DataSet_name)(
        f"./datasets/RawData", download=True, train=True, transform=ToTensor()
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

    return train_datasets


def get_model():
    ## User-defined model
    model = CNN(num_channel, num_classes, num_pixel)
    return model


def main():

    parser = argparse.ArgumentParser(description="Provide IP address")
    parser.add_argument("--ip", type=str, required=True)
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--nclients", type=int, required=True)
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    torch.manual_seed(1)

    start_time = time.time()
    train_datasets = get_data(args.nclients)
    model = get_model()
    print(
        "----------Loaded Datasets and Model----------Elapsed Time=",
        time.time() - start_time,
    )

    # read default configuration
    cfg = OmegaConf.structured(Config)
    cfg.server.host = args.ip
    print(OmegaConf.to_yaml(cfg))

    grpc_client.run_client(cfg, args.client_id, model, train_datasets[args.client_id - 1])
    print("------DONE------")


if __name__ == "__main__":
    main()
