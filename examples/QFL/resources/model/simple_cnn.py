import torch.nn as nn


class SimpleCNN(nn.Module):
    """Simple CNN model for MNIST classification - matches our original implementation"""

    def __init__(self, num_channel=1, num_classes=10, num_pixel=28):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(num_channel, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the size after convolutions and pooling
        # After conv1 + pool: 28x28 -> 14x14
        # After conv2 + pool: 14x14 -> 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # First conv block
        x = self.pool(self.relu(self.conv1(x)))

        # Second conv block
        x = self.pool(self.relu(self.conv2(x)))

        # Flatten for fully connected layers
        x = x.view(-1, 64 * 7 * 7)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
