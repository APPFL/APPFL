def get_loss():
    import torch
    import torch.nn as nn
    class MyBCELoss(nn.BCELoss):
        def forward(self, input, target):
            return super().forward(input, target.float().unsqueeze(1))
    return MyBCELoss