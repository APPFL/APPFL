import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class Baseline(nn.Module):
    """Baseline model
    We use the EfficientNets architecture that many participants in the ISIC
    competition have identified to work best.
    See here the [reference paper](https://arxiv.org/abs/1905.11946)
    Thank you to [Luke Melas-Kyriazi](https://github.com/lukemelas) for his
    [pytorch reimplementation of EfficientNets]
    (https://github.com/lukemelas/EfficientNet-PyTorch).
    """

    def __init__(self, pretrained=True, arch_name="efficientnet-b0"):
        super(Baseline, self).__init__()
        self.pretrained = pretrained
        self.base_model = (
            EfficientNet.from_pretrained(arch_name)
            if pretrained
            else EfficientNet.from_name(arch_name)
        )
        # self.base_model=torchvision.models.efficientnet_v2_s(pretrained=pretrained)
        nftrs = self.base_model._fc.in_features
        print("Number of features output by EfficientNet", nftrs)
        self.base_model._fc = nn.Linear(nftrs, 8)

    def forward(self, image):
        out = self.base_model(image)
        return out