import argparse

import numpy as np
import math
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import ToTensor

from appfl.config import *
from appfl.misc.data import *
import appfl.run_grpc_server as grpc_server
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
    parser.add_argument('--server_port', type=int, default=50051)

    args = parser.parse_args()

    num_clients = args.num_clients

    test_data_raw = torchvision.datasets.MNIST(
        "./_data", train=False, download=True, transform=ToTensor()
    )
    test_data_input = []
    test_data_label = []
    for idx in range(len(test_data_raw)):
        test_data_input.append(test_data_raw[idx][0].tolist())
        test_data_label.append(test_data_raw[idx][1])

    test_dataset = Dataset(
        torch.FloatTensor(test_data_input), torch.tensor(test_data_label)
    )

    model = CNN()

    cfg = OmegaConf.structured(Config)
    cfg.server.host = args.server_host
    cfg.server.port = args.server_port
    print(OmegaConf.to_yaml(cfg))

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    grpc_server.run_server(cfg, model, num_clients, test_dataset)


if __name__ == "__main__":
    main()
