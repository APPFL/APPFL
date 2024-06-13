import torch
import torch.nn as nn

class LongFilterCNN(nn.Module):

    def __init__(self, in_channels = 1):
        super(LongFilterCNN, self).__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels, 64, (80, 3), 1),
            nn.ReLU(),
        )

        self.conv1d = nn.Sequential(
            nn.Conv1d(64, 64, 3, 1),
            nn.Conv1d(64, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 32, 3, 1),
            nn.Conv1d(32, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            )

        self.classifier = nn.Sequential(
            nn.Linear(1920, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):

        x = self.conv2d(x)
        x = torch.squeeze(x, dim=2)
        x = self.conv1d(x)
        x = torch.flatten(x, 1)
        output = self.classifier(x)
        return output
