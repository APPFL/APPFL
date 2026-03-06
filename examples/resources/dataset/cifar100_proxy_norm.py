"""
Proxy dataset for DIMAT aggregator with CIFAR-100 normalization.
Uses the full CIFAR-100 training set with lazy transforms (applied on each
__getitem__ call) matching the original DIMAT code's trainloader.
"""

import os
import torchvision
import torchvision.transforms as transforms


def get_cifar100_proxy(**kwargs):
    """
    Return the full CIFAR-100 training set as proxy data.
    Matches the original DIMAT code which uses all 50,000 training images
    with random augmentation applied lazily on each access.
    """
    dir = os.getcwd() + "/datasets/RawData"

    cifar_mean = [0.4914, 0.4822, 0.4465]
    cifar_std = [0.2023, 0.1994, 0.2010]

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
        ]
    )

    return torchvision.datasets.CIFAR100(
        dir, download=True, train=True, transform=train_transform
    )
