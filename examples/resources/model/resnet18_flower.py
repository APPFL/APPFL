"""
ResNet18 for natural-image datasets (224x224 input).
Wraps torchvision's ResNet18 with a configurable output head.

Used for Flower102 and similar fine-grained recognition tasks where the
standard ImageNet-style architecture (7x7 conv, maxpool, 4 stages) is
appropriate.
"""

import torch.nn as nn
import torchvision.models as tv_models


def ResNet18(num_classes: int = 102, pretrained: bool = False):
    """
    ResNet18 with the final fully-connected layer replaced to match
    ``num_classes``.

    Args:
        num_classes: Number of output classes (102 for Flower102).
        pretrained:  If True, initialise the backbone with ImageNet weights.
                     Set to False for a fair comparison in FL experiments.
    """
    if pretrained:
        model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = tv_models.resnet18(weights=None)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
