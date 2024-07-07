import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layer (input channels = 1, output channels = 32, kernel size = 3x3)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        # Convolutional layer (input channels = 32, output channels = 64, kernel size = 3x3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        # Maxpooling layer (kernel size = 2x2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layer (64 * 12 * 12 -> 128)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        # Fully connected layer (128 -> 10)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
if __name__ == '__main__':
    # Create an instance of the CNN
    model = SimpleCNN()

    # Calculate the total number of parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Print the number of parameters
    print(f'Total number of parameters: {count_parameters(model)}')

    # Calculate the model size in bytes and convert to MB
    def get_model_size(model):
        param_size = 0
        for param in model.parameters():
            param_size += param.numel() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.numel() * buffer.element_size()
        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb

    # Print the model size
    print(f'Model size: {get_model_size(model):.2f} MB')

