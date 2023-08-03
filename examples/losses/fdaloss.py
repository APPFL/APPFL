import torch.nn as nn
import torch

class BCELoss(nn.Module):
    '''Cross Entroy Loss'''
    def __init__(self):
        super(BCELoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, prediction, target):
        target = target.unsqueeze(1)
        target = target.type_as(prediction)
        return self.criterion(prediction, target)

class MTLLoss(nn.Module):
    '''Cross Entroy Loss'''
    def __init__(self):
        super(MTLLoss, self).__init__()
        criterion_covid = torch.nn.BCEWithLogitsLoss(reduction='mean')
        criterion_race = torch.nn.CrossEntropyLoss()
        criterion_sex = torch.nn.CrossEntropyLoss()
        criterion_age = torch.nn.CrossEntropyLoss()
        self.loss_fn = [criterion_covid, criterion_race, criterion_sex, criterion_age]
        # self.target_id = target_id

    def forward(self, prediction, target, target_id=-1, client_id=-1):
        pred_main = prediction[0]
        labels = target[0]
        labels = labels.unsqueeze(1)
        labels = labels.type_as(pred_main)    
        loss = self.loss_fn[0](pred_main, labels)
        if client_id >= 0 and target_id >= 0: # and client_id != target_id:
            for idx, c in enumerate(self.loss_fn[1:]):
                loss += c(prediction[idx+1], target[idx+1])
        
        return loss