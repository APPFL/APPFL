import torch

def mse_loss(pred, y):
    return torch.nn.MSELoss()(pred.float(), y.unsqueeze(-1).float())