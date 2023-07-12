import torch
import torch.nn as nn
import math


class CNN(nn.Module):
    def __init__(self, num_channel, num_classes, num_pixel):
        super().__init__()
        self.conv1 = nn.Conv2d(
            num_channel, 32, kernel_size=5, padding=0, stride=1, bias=True
        )
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        self.act = nn.ReLU(inplace=True)

        ###
        ### X_out = floor{ 1 + (X_in + 2*padding - dilation*(kernel_size-1) - 1)/stride }
        ###
        X = num_pixel
        X = math.floor(1 + (X + 2 * 0 - 1 * (5 - 1) - 1) / 1)
        X = X / 2
        X = math.floor(1 + (X + 2 * 0 - 1 * (5 - 1) - 1) / 1)
        X = X / 2
        X = int(X)

        self.fc1 = nn.Linear(64 * X * X, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        self.conv1.bias = nn.Parameter(self.conv1.bias.float())
        self.conv2.bias = nn.Parameter(self.conv2.bias.float())
        x = x.float()
        x = self.act(self.conv1(x))
        x = self.maxpool(x)
        x = self.act(self.conv2(x))
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


class ComplexCNN(nn.Module):
    def __init__(self, num_channel, num_classes, num_pixel):
        super().__init__()
        self.conv1 = nn.Conv2d(
            num_channel, 128, kernel_size=3, padding=1, stride=1, bias=True
        )
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv4 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, stride=1, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        self.act = nn.ReLU(inplace=True)

        X = num_pixel
        self.fc1 = nn.Linear(1024 * X * X, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.maxpool(x)
        x = self.act(self.conv2(x))
        x = self.maxpool(x)
        x = self.act(self.conv3(x))
        x = self.maxpool(x)
        x = self.act(self.conv4(x))
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x
