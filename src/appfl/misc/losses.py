import torch

def mse_loss(pred, y):
    return torch.nn.MSELoss()(pred, y.unsqueeze(-1))

def get_loss_func(loss_name = "CrossEntropy"):
    if loss_name == "CrossEntropy":
        return torch.nn.CrossEntropyLoss()
    elif loss_name == "MSE":
        return mse_loss
    else:
        raise Exception("Loss function is not supported")

    

