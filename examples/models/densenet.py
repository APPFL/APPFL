def get_model():
    import torchvision
    import torch.nn as nn
    class DenseNet121(nn.Module):
        """
        DenseNet121 model with additional Sigmoid layer for classification
        """
        def __init__(self, num_output):
            super(DenseNet121, self).__init__()
            self.densenet121 = torchvision.models.densenet121(pretrained = True)
            num_features = self.densenet121.classifier.in_features
            self.densenet121.classifier = nn.Sequential(
                nn.Linear(num_features, num_output),
            )
        def forward(self, x):
            x = self.densenet121(x)
            return x
    ## User-defined model
    return DenseNet121

