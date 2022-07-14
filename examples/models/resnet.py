from torchvision import *

def resnet18(num_classes=None):
    if num_classes is not None:
        return models.resnet18(pretrained=False, num_classes=num_classes)
    else:
        return models.resnet18(pretrained=True)
