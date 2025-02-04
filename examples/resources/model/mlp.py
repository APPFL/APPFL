import torch
import torch.nn as nn



class MLP(nn.Module):
    def __init__(self, input_size,num_classes):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, num_classes)  # Output a single value for binary classification
        
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.sigmoid = nn.Sigmoid()  # Add sigmoid activation for binary classification
        
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.act(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)  # Apply sigmoid to output
        return x



class Baseline(nn.Module):
    def __init__(self, input_dim=13, output_dim=1):
        super(Baseline, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))