def get_model():
    import torchvision
    import torch.nn as nn
    class ResNet(nn.Module):
        """
        DenseNet121 model with additional Sigmoid layer for classification
        """
        def __init__(self, num_output):
            super(ResNet, self).__init__()
            self.ResNet50 = torchvision.models.resnet50(pretrained=False)
            self.ResNet50.fc = nn.Sequential(
            nn.Linear(2048, num_output)
        )
        def forward(self, x):
            x = self.ResNet50(x)
            return x
    ## User-defined model
    return ResNet
