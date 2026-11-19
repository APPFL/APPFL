import torch.nn as nn


class CELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, prediction, target):
        return self.loss(prediction, target)
