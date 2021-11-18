import sys

sys.path.append("..")

## User-defined model

import torch
import torch.nn as nn
import math


class CNN1(nn.Module):
    def __init__(self, in_features, num_classes, pixel):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_features, 32, kernel_size=5, padding=0, stride=1, bias=True
        )
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        self.act = nn.ReLU(inplace=True)

        ###################################################################################
        #### X_out = floor{ 1 + (X_in + 2*padding - dilation*(kernel_size-1) - 1)/stride }
        ###################################################################################
        X = pixel
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


model = CNN1(1, 10, 28)


## User-defined datasets

import torchvision
from torchvision.transforms import ToTensor

train_data = torchvision.datasets.MNIST(
    f"./datasets",
    download=True,
    train=True,
    transform=ToTensor(),
)
test_data = torchvision.datasets.MNIST(
    f"./datasets",
    download=True,
    train=False,
    transform=ToTensor(),
)


## train

import appfl.run_idea as fl
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../appfl/config", config_name="config")
def main(cfg: DictConfig):

    fl.run_serial(cfg, model, train_data, test_data)


if __name__ == "__main__":
    main()
