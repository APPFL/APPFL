import torch
import torch.nn as nn
import torch.nn.functional as F
import math
 
class CNN2(nn.Module):
    def __init__(self, in_features, num_classes, pixel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, 6, 5)  ## in_channels, out_channels, kernel_size
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        ################################################################################### 
        #### X_out = floor{ 1 + (X_in + 2*padding - dilation*(kernel_size-1) - 1)/stride }
        ################################################################################### 
        X = pixel
        X = math.floor( 1+(X + 2*0 - 1*(5-1)-1)/1 )
        X = X/2
        X = math.floor( 1+(X + 2*0 - 1*(5-1)-1)/1 )
        X = X/2
        X = int(X)

        self.fc1 = nn.Linear(16 * X * X, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)        

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
