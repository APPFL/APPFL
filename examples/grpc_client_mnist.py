import argparse

import numpy as np
import math
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import ToTensor

from appfl.config import *
from appfl.misc.data import *
import appfl.run_grpc_client as grpc_client
import sys
import logging


class CNN(nn.Module):
    def __init__(self, num_channel=1, num_classes=10, num_pixel=28):
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clients', type=int, default=1)
    parser.add_argument('--server_host', type=str, default="localhost")

    args = parser.parse_args()
    ## num_clients should be the same as which was setted in server
    num_clients = args.num_clients
    train_data_raw = torchvision.datasets.MNIST(
        "./_data", train=True, download=True, transform=ToTensor()
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

    model = CNN()

    cfg = OmegaConf.structured(Config)
    cfg.server.host = args.server_host
    print(OmegaConf.to_yaml(cfg))

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    # If you have more than 1 client, then change the param "1" and "train_datasets[0]" to "i" and "train_datasets[i-1]"
    # which "i" stands for the i-th client ,so the following line should be different according to the client.
    grpc_client.run_client(cfg, 0, model, train_datasets[0])
    # grpc_client.run_client(cfg, 2, model, train_datasets[1])
    # grpc_client.run_client(cfg, 3, model, train_datasets[2])
    # grpc_client.run_client(cfg, 4, model, train_datasets[3])
    # ...


if __name__ == '__main__':
    main()
