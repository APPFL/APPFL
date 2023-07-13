def get_model():
    import torchvision
    import torch.nn as nn

    class ResNet(nn.Module):
        """
        ResNet model with a modified fc layer for classification
        """

        def __init__(self, num_output):
            super(ResNet, self).__init__()
            self.ResNet18 = torchvision.models.resnet18(pretrained=True)
            self.ResNet18.fc = nn.Sequential(nn.Linear(512, num_output))
            # Freeze all BN layers
            def get_layer(model, name):
                layer = model
                for attr in name.split("."):
                    layer = getattr(layer, attr)
                return layer
            
            def set_layer(model, name, layer):
                try:
                    attrs, name = name.rsplit(".", 1)
                    model = get_layer(model, attrs)
                except ValueError:
                    pass
                setattr(model, name, layer)

            for name, module in self.ResNet18.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    # Get current bn layer
                    bn = get_layer(self.ResNet18, name)
                    # Create new gn layer
                    gn = nn.GroupNorm(1, bn.num_features)
                    # Assign gn
                    set_layer(self.ResNet18, name, gn)
        
        def forward(self, x):
            x = self.ResNet18(x)
            return x

    ## User-defined model
    return ResNet
