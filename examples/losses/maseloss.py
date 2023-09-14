import torch
import torch.nn as nn

class MASELoss(nn.Module):
    '''Mean Absolute Scaled Error Loss'''
    def __init__(self):
        super(MASELoss, self).__init__()

    def forward(self, prediction, target):
        numerator = torch.mean( torch.abs(prediction-target) )
        denominator = torch.mean( torch.abs(torch.diff(target),n=1) )
        return torch.divide(numerator,denominator)