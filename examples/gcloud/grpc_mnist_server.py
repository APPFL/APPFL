import sys
import time
import logging
import argparse
import math

import torch
import torch.nn as nn

from appfl.config import *
import appfl.run_grpc_server as grpc_server

num_channel = 1  # 1 if gray, 3 if color
num_classes = 10  # number of the image classes
num_pixel = 28  # image size = (num_pixel, num_pixel)


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
    parser.add_argument("--nclients", type=int, required=True)
    parser.add_argument("--logging", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=eval("logging." + args.logging))
    torch.manual_seed(1)

    start_time = time.time()
    model = CNN(num_channel, num_classes, num_pixel)

    logger = logging.getLogger(__name__)
    logger.info(f"----------Loaded Model----------Elapsed Time={time.time() - start_time}")

    # read default configuration
    cfg = OmegaConf.structured(Config)

    grpc_server.run_server(cfg, model, args.nclients)


if __name__ == "__main__":
    main()
