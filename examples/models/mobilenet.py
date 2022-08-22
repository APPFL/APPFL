def get_model():
    import torchvision
    import torch.nn as nn
    class MobileNet(nn.Module):
        """
        DenseNet121 model with additional Sigmoid layer for classification
        """
        def __init__(self, num_output):
            super(MobileNet, self).__init__()
            #mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
            self.MobileNet = torchvision.models.mobilenet_v2()
            self.MobileNet.classifier[1] = nn.Linear(in_features=self.MobileNet.classifier[1].in_features, out_features=num_output)
        
        def forward(self, x):
            x = self.MobileNet(x)
            return x
    ## User-defined model
    return MobileNet
