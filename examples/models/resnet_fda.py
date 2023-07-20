from torch import nn
from torchvision import models
import torch
import torch.nn.functional as F


class ResNetClassifier(nn.Module):
    """
    Pre-trained ResNet on ImageNet with one added hidden layer, normalization,
    and activation.
    """
    def __init__(self, hidden_size=512, resnet='resnet152', pretrained=True):
        super().__init__()

        if resnet == 'resnet18':
            self.resnet = models.resnet18(pretrained=pretrained)
        elif resnet == 'resnet34':
            self.resnet = models.resnet34(pretrained=pretrained)
        elif resnet == 'resnet50':
            self.resnet = models.resnet50(pretrained=pretrained)
        elif resnet == 'resnet101':
            self.resnet = models.resnet101(pretrained=pretrained)
        elif resnet == 'resnet152':
            self.resnet = models.resnet152(pretrained=pretrained)
        elif resnet == 'vgg':
            self.resnet = models.vgg16(pretrained=pretrained)

        if resnet == 'vgg':
            num_feats = self.resnet.classifier[6].in_features
            self.resnet.classifier[6] = nn.Linear(num_feats, hidden_size)
        else:
            num_feats = self.resnet.fc.in_features
            self.resnet.fc = nn.Linear(num_feats, hidden_size)

        
        self.linear = nn.Linear(hidden_size, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):

        x = self.leaky_relu(self.resnet(x))
        x = self.leaky_relu(self.linear(x))

        # return torch.sigmoid(x)
        return x

class ResnetMultiTaskNet(nn.Module):
    """
    Pre-trained ResNet on ImageNet with one added hidden layer, normalization,
    and activation. Change it to support MLT setting with self-determined branches

    num_classes: a list of ints to define how many outputs we need for each task.
    """
    def __init__(self, pretrained=True, frozen_feature_layers=False, resnet='resnet152', hidden_size=512, num_classes=[1,3,2,10]):
        super().__init__()
        if resnet == 'resnet18':
            self.resnet = models.resnet18(pretrained=True)
        elif resnet == 'resnet34':
            self.resnet = models.resnet34(pretrained=True)
        elif resnet == 'resnet50':
            self.resnet = models.resnet50(pretrained=True)
        elif resnet == 'resnet101':
            self.resnet = models.resnet101(pretrained=True)
        elif resnet == 'resnet152':
            self.resnet = models.resnet152(pretrained=True)
        elif resnet == 'vgg':
            self.resnet = models.vgg16(pretrained=True)

        # self.resnet.fc = nn.Linear(num_feats, hidden_size)
        self.is_frozen = frozen_feature_layers
        # here we get all the modules(layers) before the fc layer at the end
        # note that currently at pytorch 1.0 the named_children() is not supported
        # and using that instead of children() will fail with an error
        # self.features = nn.ModuleList(self.resnet.children())[:-1]
        # this is needed because, nn.ModuleList doesnt implement forward()
        # so you cant do sth like self.features(images). therefore we use 
        # nn.Sequential and since sequential doesnt accept lists, we 
        # unpack all items and send them like this
        # self.features = nn.Sequential(*self.features)

        if frozen_feature_layers:
            self.freeze_feature_layers()

        # now lets add our new layers 
        if resnet == 'vgg':
            num_feats = self.resnet.classifier[6].in_features
            self.resnet.classifier[6] = nn.Linear(num_feats, hidden_size)
        else:
            num_feats = self.resnet.fc.in_features
            self.resnet.fc = nn.Linear(num_feats, hidden_size)
        # it helps with performance. you can play with it
        # create more layers, play/experiment with them. 
        # self.fc0 = nn.Linear(in_features, hidden_size)
        # self.bn_pu = nn.BatchNorm1d(hidden_size, eps = 1e-5)
        # our five new heads for 5 tasks we have at hand!
        self.fc_out = []
        for n_cls in num_classes:
            self.fc_out.append(nn.Linear(hidden_size, n_cls))
        self.fc_out = nn.ModuleList(self.fc_out)
        # self.fc_covid = nn.Linear(hidden_size, 1) # binary classification
        # self.fc_race = nn.Linear(hidden_size, 4) # {'White':0, 'Black or African American':1, 'Asian':2, 'Other':3} 
        # self.fc_region = nn.Linear(hidden_size, 4) 
        # self.fc_fighting = nn.Linear(hidden_size, 3)
        # self.fc_alignment = nn.Linear(hidden_size, 9)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        # initialize all fc layers to xavier
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         torch.nn.init.xavier_normal_(m.weight, gain = 1)


    def forward(self, input_imgs):
        # output = self.resnet(input_imgs)
        # output = output.view(input_imgs.size(0), -1)
        # output = self.bn_pu(F.relu(output))
        x = self.leaky_relu(self.resnet(input_imgs))
        # since color is multi label we should use sigmoid
        # but since we want a numerical stable one, we use
        # nn.BCEWithLogitsloss, as a loss which itself applies sigmoid
        # and thus accepts logits. so we wont use sigmoid here for that matter
        # its much stabler than sigmoid+BCE
        preds = []
        for i in range(len(self.fc_out)):
            pred = self.leaky_relu(self.fc_out[i](x))
            # if i > 0:
            # pred =  F.log_softmax(pred, dim = 1)
            preds.append(pred)


        # prd_covid = self.leaky_relu(self.fc_covid(x))
        # prd_race = F.log_softmax(self.leaky_relu(self.fc_race(x)), dim=1)
        # prd_region = self.fc_region(output)
        # prd_fighting = self.fc_fighting(output)
        # prd_alingment = self.fc_alignment(output)
        
        return preds
    
    def _set_freeze_(self, status):
        for n,p in self.features.named_parameters():
            p.requires_grad = status
        # for m in self.features.children():
        #     for p in m.parameters():
        #         p.requires_grad=status    


    def freeze_feature_layers(self):
        self._set_freeze_(False)

    def unfreeze_feature_layers(self):
        self._set_freeze_(True)
    

    