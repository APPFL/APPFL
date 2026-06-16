"""MLP matching AFL-Lib's model/mlp.py (mlp_mnist) for a controlled comparison.

Architecture identical to AFL-Lib: 784 -> 512 -> 256 -> num_classes, ReLU after
every layer (including the output layer, as in AFL-Lib). Input is flattened.
"""

import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc(x))  # AFL-Lib applies ReLU on the output too
        return x
