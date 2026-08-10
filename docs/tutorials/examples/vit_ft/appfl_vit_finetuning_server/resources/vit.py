import torch
from torchvision.models import ViT_B_16_Weights, vit_b_16


def get_vit():
    """
    Return a pretrained ViT with all layers frozen except output head.
    """

    # Instantiate a pre-trained ViT-B on ImageNet
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    in_features = model.heads[-1].in_features
    model.heads[-1] = torch.nn.Linear(in_features, 2)

    # Disable gradients for everything
    model.requires_grad_(False)
    # Now enable just for output head
    model.heads.requires_grad_(True)

    return model
