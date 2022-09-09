import torch

def mse_loss(pred, y):
    return torch.nn.MSELoss()(pred, y.unsqueeze(-1))