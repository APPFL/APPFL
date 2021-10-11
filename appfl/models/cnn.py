import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features,
                               32,
                               kernel_size=5,
                               padding=0,
                               stride=1,
                               bias=True)
        self.conv2 = nn.Conv2d(32,
                               64,
                               kernel_size=5,
                               padding=0,
                               stride=1,
                               bias=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self.act = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.maxpool(x)
        x = self.act(self.conv2(x))
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x