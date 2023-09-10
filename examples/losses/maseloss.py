import torch
import torch.nn as nn

class MASELoss(nn.Module):
    '''Mean Absolute Scaled Error Loss'''
    def __init__(self):
        super(MASELoss, self).__init__()

    def forward(self, prediction, target):
        
        # do sanity checks
        
        if len(prediction.shape) != len(target.shape):
            raise TypeError('Input and target to MASE loss have different number of dimensions.')
        for idx,_ in enumerate(prediction.shape):
            if prediction.shape[idx] != target.shape[idx]:
                raise TypeError('Input and target to MASE loss have same num. of dimensions but different shapes.')
            
        if len(prediction.shape) > 2:
            raise ValueError('MASE error only supports 1D time series.')
        
        # calculate MASE error according to batched or unbatched
            
        if len(prediction.shape) == 1: # unbatched
            if prediction.shape[0] <= 1:
                raise ValueError('For calculating MASE, entries must be time series with 2 or more indices.')
            num = torch.mean( torch.abs(prediction-target) )
            den = torch.mean( torch.abs(target.diff(n=1)) )
            return torch.div(num,den)
        else: # batched
            if prediction.shape[1] <= 1:
                raise ValueError('For calculating MASE, entries must be time series with 2 or more indices.')
            num = torch.mean( torch.abs(prediction-target), dim=1 )
            den = torch.mean( torch.abs(target.diff(n=1,dim=1)), dim=1 )
            return torch.div(num,den).mean() # equivalent to reduction='mean'