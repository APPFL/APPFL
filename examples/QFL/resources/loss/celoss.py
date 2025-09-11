import torch
import torch.nn as nn

class CELoss(nn.Module):
    """Cross-entropy loss for classification"""
    
    def __init__(self):
        super(CELoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, input, target):
        return self.loss_fn(input, target) 