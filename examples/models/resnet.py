from torchvision import *
import torch

def resnet18(num_classes=None):
    if num_classes is not None:
        model = models.resnet18(pretrained=False, num_classes=num_classes)
        # for name, param in model.named_parameters():
        #     if len(param.shape) > 1:
        #         torch.nn.init.xavier_normal_(model.state_dict()[name])
        return model
    else:
        return models.resnet18(pretrained=True)
