import sys
import time
import logging
import argparse
import math

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import ToTensor

from appfl.config import *
from appfl.misc.data import *
import appfl.run_grpc_client as grpc_client

DataSet_name = "MNIST"
num_channel = 1  # 1 if gray, 3 if color
num_classes = 10  # number of the image classes
num_pixel = 28  # image size = (num_pixel, num_pixel)


def get_data(num_clients: int):

    # training data for multiple clients
    train_data_raw = eval("torchvision.datasets." + DataSet_name)(
        f"./_data", download=True, train=True, transform=ToTensor()
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


class CNN(nn.Module):
    def __init__(self, num_channel, num_classes, num_pixel):
        super().__init__()
        self.conv1 = nn.Conv2d(
            num_channel, 32, kernel_size=5, padding=0, stride=1, bias=True
        )
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        self.act = nn.ReLU(inplace=True)

        X = num_pixel
        X = math.floor(1 + (X + 2 * 0 - 1 * (5 - 1) - 1) / 1)
        X = X / 2
        X = math.floor(1 + (X + 2 * 0 - 1 * (5 - 1) - 1) / 1)
        X = X / 2
        X = int(X)

        self.fc1 = nn.Linear(64 * X * X, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.maxpool(x)
        x = self.act(self.conv2(x))
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


def main():

    parser = argparse.ArgumentParser(description="Provide the configuration")
    parser.add_argument("--host", type=str, required=True)
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--use_tls", type=bool, default=False)
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--nclients", type=int, required=True)
    parser.add_argument("--logging", type=str, default="DEBUG")
    parser.add_argument("--api_key", default=None)
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=eval("logging." + args.logging))
    torch.manual_seed(1)

    start_time = time.time()
    train_datasets = get_data(args.nclients)

    """ Configuration """     
    cfg = OmegaConf.structured(Config)

    """ get model         
        Note: 
        "torch.nn.CrossEntropyLoss()" should be used for the multiclass classfication problem with more than 2 classes.                
    """
    # read default configuration
    cfg = OmegaConf.structured(Config)
    model = CNN(num_channel, num_classes, num_pixel)
    loss_fn = torch.nn.CrossEntropyLoss()
    

    logger = logging.getLogger(__name__)
    logger.info(
        f"----------Loaded Datasets and Model----------Elapsed Time={time.time() - start_time}"
    )
 
    cfg.server.host = args.host
    cfg.server.port = args.port
    cfg.server.use_tls = args.use_tls
    cfg.server.api_key = args.api_key

    logger.debug(OmegaConf.to_yaml(cfg)) 
    
    grpc_client.run_client(
        cfg, args.client_id, model, loss_fn, train_datasets[args.client_id]
    )
    logger.info("------DONE------")


if __name__ == "__main__":
    main()
