"""
Simple loss function for TES testing
"""

import torch
import torch.nn as nn


class SimpleLoss(nn.Module):
    """Simple cross-entropy loss for binary classification"""
    
    def __init__(self):
        super(SimpleLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, outputs, targets):
        return self.loss_fn(outputs, targets)