import torch.nn as nn


class TinyNet(nn.Module):
    """Four inputs, two classes, no buffers. Small enough that a whole DFL round is instant."""

    def __init__(self, num_features: int = 4, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.fc(x)


class TinyNetWithNorm(nn.Module):
    """The same, plus a BatchNorm -- so its state carries a non-float buffer.

    `num_batches_tracked` is an int64 counter, and averaging it is the case no model in
    `examples/` exercises: the example CNN is Conv/Linear only, so a dtype mistake in the
    aggregator would never show up there.
    """

    def __init__(self, num_features: int = 4, num_classes: int = 2):
        super().__init__()
        self.norm = nn.BatchNorm1d(num_features)
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.fc(self.norm(x))
