import torch 

def resnet18(num_classes=10):
    return torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
