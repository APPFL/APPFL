import torch

from appfl.sim.models.model_utils import Lambda



class M5(torch.nn.Module):
    def __init__(self, in_channels, hidden_size, num_classes):
        super(M5, self).__init__()
        self.in_channels = in_channels
        self.num_hiddens = hidden_size
        self.num_classes = num_classes

        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(self.in_channels, self.num_hiddens, kernel_size=80, stride=4),
            torch.nn.BatchNorm1d(self.num_hiddens),
            torch.nn.ReLU(True),
            torch.nn.MaxPool1d(4),
            torch.nn.Conv1d(self.num_hiddens, self.num_hiddens, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(self.num_hiddens),
            torch.nn.ReLU(True),
            torch.nn.MaxPool1d(4),
            torch.nn.Conv1d(self.num_hiddens, self.num_hiddens * 2, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(self.num_hiddens * 2),
            torch.nn.ReLU(True),
            torch.nn.MaxPool1d(4),
            torch.nn.Conv1d(self.num_hiddens * 2, self.num_hiddens * 4, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(self.num_hiddens * 4),
            torch.nn.ReLU(True),
            torch.nn.MaxPool1d(4),
        )
        self.classifier = torch.nn.Sequential(
            Lambda(lambda x: torch.nn.functional.avg_pool1d(x, x.shape[-1]).permute(0, 2, 1)),
            torch.nn.Linear(self.num_hiddens * 4, self.num_classes),
            torch.nn.Flatten()
            )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x