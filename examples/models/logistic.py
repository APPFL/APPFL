import torch
import torch.nn as nn
from torch.autograd import Variable

class LogisticRegression(torch.nn.Module):
    def __init__(self, num_channel, num_classes, num_pixel):
        super(LogisticRegression, self).__init__()
        self.size = num_channel*num_pixel*num_pixel                
        self.linear = torch.nn.Linear(self.size, num_classes)
    def forward(self, x):
        # x = Variable(x.view(-1, self.size))        
        batch_size, channels, width, height = x.size()
        x=x.view(batch_size,-1)
        outputs = self.linear(x)
        return outputs
