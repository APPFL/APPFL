import math

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import nn


class LightningCNN(pl.LightningModule):
    def __init__(self, num_channel=1, num_classes=10, num_pixel=28, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.conv1 = nn.Conv2d(num_channel, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        self.act = nn.ReLU(inplace=False)

        X = num_pixel
        X = math.floor(1 + (X - 1 * (5 - 1) - 1) / 1)
        X = X / 2
        X = math.floor(1 + (X - 1 * (5 - 1) - 1) / 1)
        X = int(X / 2)

        self.fc1 = nn.Linear(64 * X * X, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.maxpool(self.act(self.conv1(x)))
        x = self.maxpool(self.act(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        return self.fc2(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_accuracy", acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_accuracy", acc, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
