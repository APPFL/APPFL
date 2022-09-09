def get_model():
    import torchvision
    import torch.nn as nn
    class ResNet(nn.Module):
        """
        DenseNet121 model with additional Sigmoid layer for classification
        """
        def __init__(self, num_output):
            super(ResNet, self).__init__()
            self.ResNet18 = torchvision.models.resnet18(pretrained=False)
            self.ResNet18.fc = nn.Sequential(
            nn.Linear(512, num_output)
        )
        def forward(self, x):
            x = self.ResNet18(x)
            return x
    ## User-defined model
    return ResNet

