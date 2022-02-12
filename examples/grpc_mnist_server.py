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

DataSet_name = "MNIST"
num_channel = 1  # 1 if gray, 3 if color
num_classes = 10  # number of the image classes
num_pixel = 28  # image size = (num_pixel, num_pixel)


def get_data():

    # test data for a server
    test_data_raw = eval("torchvision.datasets." + DataSet_name)(
        f"./datasets/RawData", download=True, train=False, transform=ToTensor()
    )

    test_data_input = []
    test_data_label = []
    for idx in range(len(test_data_raw)):
        test_data_input.append(test_data_raw[idx][0].tolist())
        test_data_label.append(test_data_raw[idx][1])

    test_dataset = Dataset(
        torch.FloatTensor(test_data_input), torch.tensor(test_data_label)
    )

    return test_dataset


def get_model():
    ## User-defined model
    model = CNN(num_channel, num_classes, num_pixel)
    return model


def main():

    parser = argparse.ArgumentParser(description="Provide IP address")
    parser.add_argument("--ip", type=str, required=True)
    parser.add_argument("--nclients", type=int, required=True)
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    torch.manual_seed(1)

    start_time = time.time()
    test_dataset = get_data()
    model = get_model()
    print(
        "----------Loaded Datasets and Model----------Elapsed Time=",
        time.time() - start_time,
    )

    # read default configuration
    cfg = OmegaConf.structured(Config)
    cfg.server.host = args.ip
    print(OmegaConf.to_yaml(cfg))

    grpc_server.run_server(
        cfg, 0, model, test_dataset, args.nclients, DataSet_name
    )


if __name__ == "__main__":
    main()
