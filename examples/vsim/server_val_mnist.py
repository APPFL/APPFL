"""Server-side validation dataset for the vsim simulator: the full MNIST test set.

Used via `server_configs.val_data_configs` so the driver can evaluate the GLOBAL model
on a common held-out set (true convergence curve), independent of per-client non-IID val.
Preprocessing matches resources/dataset/mnist_dataset.py (ToTensor).
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms
from appfl.misc.data import Dataset


def get_mnist_test(**kwargs):
    dir = os.getcwd() + "/datasets/RawData"
    test_raw = torchvision.datasets.MNIST(
        dir, download=True, train=False, transform=transforms.ToTensor()
    )
    xs, ys = [], []
    for idx in range(len(test_raw)):
        xs.append(test_raw[idx][0].tolist())
        ys.append(test_raw[idx][1])
    return Dataset(torch.FloatTensor(xs), torch.tensor(ys))
