import torch

from appfl.sim.models.model_utils import DANetBlock



class DANet(torch.nn.Module):
    def __init__(self, in_features, num_layers, hidden_size, num_classes, dropout, B):
        super(DANet, self).__init__()
        params = {
            'fix_input_dim': in_features, 'drop_rate': dropout,
            'base_outdim': hidden_size, 
            'k': 3, 
            'virtual_batch_size': B
        }
        self.features = torch.nn.ModuleList([DANetBlock(in_features, **params)])
        for _ in range((num_layers // 2) - 1):
            self.features.append(DANetBlock(hidden_size, **params))
        else:
            self.features.append(torch.nn.Dropout(0.1))
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        out = self.features[0](x)
        for i in range(1, len(self.features) - 1):
            out = self.features[i](x, out)
        out = self.classifier(out)
        return out