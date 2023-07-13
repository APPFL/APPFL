def get_model():
    import torchvision
    import torch.nn as nn
    from torch.autograd import Function


    class ReverseLayerF(Function):
        @staticmethod
        def forward(ctx, x, alpha):
            ctx.alpha = alpha

            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output):
            output = grad_output.neg() * ctx.alpha

            return output, None
        
    class ResNetDDA(nn.Module):
        """
        DenseNet121 model with additional Sigmoid layer for classification
        """

        def __init__(self, num_output, num_domains):
            super(ResNetDDA, self).__init__()
            self.ResNet18 = torchvision.models.resnet18(pretrained=True)
            self.ResNet18 = nn.Sequential(*list(self.ResNet18.children())[:-1])

            self.class_fc = nn.Sequential(nn.Linear(512, num_output))
            self.domain_fc = nn.Sequential(nn.Linear(512, num_domains))

            self.alpha = 1.0

        def forward(self, x):
            feat = self.ResNet18(x)
            reverse_feat = ReverseLayerF.apply(feat, self.alpha)
            class_output = self.class_fc(feat)
            domain_output = self.domain_fc(reverse_feat)
            return (class_output, domain_output)
    # import ipdb; ipdb.set_trace()
    ## User-defined model
    return ResNetDDA
