import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, 6, 5)  ## in_channels, out_channels, kernel_size
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 1*28*28 -> 16*4*4
        self.fc1 = nn.Linear(16 * 5 * 5, 120)    # 3*32*32 -> 16*5*5
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
