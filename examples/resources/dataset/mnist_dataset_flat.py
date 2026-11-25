import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
from appfl.misc.data import (
    cache_per_process,
    iid_partition,
    class_noniid_partition,
    dirichlet_noniid_partition,
)


class NumpyDataset:
    """Minimal dataset wrapper holding (X, y) numpy arrays.

    Supports len() and indexing so ClientAgent.get_sample_size() works correctly,
    and SklearnTrainer._to_numpy() can extract arrays without a DataLoader.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_test_arrays(dir: str):
    """Return the MNIST test set as flattened (X, y) numpy arrays."""
    test_raw = torchvision.datasets.MNIST(
        dir, download=True, train=False, transform=transforms.ToTensor()
    )
    test_X = np.stack([test_raw[i][0].numpy().ravel() for i in range(len(test_raw))])
    test_y = np.array([test_raw[i][1] for i in range(len(test_raw))])
    return test_X, test_y


def partition_train_dataset(
    dir: str, num_clients: int, partition_strategy: str, **kwargs
):
    """Split the MNIST training set into one tensor dataset per client."""
    train_raw = torchvision.datasets.MNIST(
        dir, download=False, train=True, transform=transforms.ToTensor()
    )

    if partition_strategy == "iid":
        return iid_partition(train_raw, num_clients)
    elif partition_strategy == "class_noniid":
        return class_noniid_partition(train_raw, num_clients, **kwargs)
    elif partition_strategy == "dirichlet_noniid":
        return dirichlet_noniid_partition(train_raw, num_clients, **kwargs)
    else:
        raise ValueError(f"Unknown partition strategy: {partition_strategy!r}")


def get_mnist_flat(
    num_clients: int,
    client_id: int,
    partition_strategy: str = "iid",
    **kwargs,
):
    """
    Return flattened MNIST data (784-dim numpy arrays) for a given client.

    The test set and the partition are built once per process and shared by all
    the clients created in it.

    Args:
        num_clients: total number of FL clients.
        client_id: zero-based index of this client.
        partition_strategy: one of "iid", "class_noniid", or "dirichlet_noniid".
        **kwargs: forwarded to the partition helper.

    Returns:
        (train_dataset, test_dataset) as NumpyDataset instances.
    """
    dir = os.getcwd() + "/datasets/RawData"

    # --- test set ---
    test_X, test_y = cache_per_process(
        "mnist_dataset_flat.test", load_test_arrays, dir=dir
    )

    # --- training set ---
    train_datasets = cache_per_process(
        "mnist_dataset_flat.train",
        partition_train_dataset,
        dir=dir,
        num_clients=num_clients,
        partition_strategy=partition_strategy,
        **kwargs,
    )

    client_raw = train_datasets[client_id]
    train_X = np.stack(
        [client_raw[i][0].numpy().ravel() for i in range(len(client_raw))]
    )
    train_y = np.array([client_raw[i][1] for i in range(len(client_raw))])

    return NumpyDataset(train_X, train_y), NumpyDataset(test_X, test_y)
