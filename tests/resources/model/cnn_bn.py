import torch
import torch.nn as nn


class CNNBN(nn.Module):
    """A small CNN with batch norm, so its state holds float buffers and int64
    counters (``num_batches_tracked``) between its parameters."""

    def __init__(self, num_channel, num_classes, num_pixel):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channel, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * (num_pixel // 4) ** 2, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        return self.fc(torch.flatten(x, 1))
