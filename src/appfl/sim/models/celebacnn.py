import torch



class CelebACNN(torch.nn.Module): # McMahan et al., 2016; 1,663,370 parameters
    def __init__(self, in_channels, hidden_size, num_classes):
        super(CelebACNN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_size
        self.num_classes = num_classes
        
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.in_channels, out_channels=self.hidden_channels, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            torch.nn.GroupNorm(num_groups=self.hidden_channels, num_channels=self.hidden_channels),
            torch.nn.MaxPool2d(kernel_size=(2, 2), padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            torch.nn.GroupNorm(num_groups=self.hidden_channels // 2, num_channels=self.hidden_channels),
            torch.nn.MaxPool2d(kernel_size=(2, 2), padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            torch.nn.GroupNorm(num_groups=self.hidden_channels // 4, num_channels=self.hidden_channels),
            torch.nn.MaxPool2d(kernel_size=(2, 2), padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            torch.nn.GroupNorm(num_groups=self.hidden_channels // 8, num_channels=self.hidden_channels),
            torch.nn.MaxPool2d(kernel_size=(2, 2), padding=1),
            torch.nn.ReLU()
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((5, 5)),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=self.hidden_channels * (5 * 5), out_features=self.num_classes, bias=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
