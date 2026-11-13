import os
import torch
import torchvision
import torchvision.transforms as transforms
from appfl.misc.data import (
    Dataset,
    cache_per_process,
    iid_partition,
    class_noniid_partition,
    dirichlet_noniid_partition,
)


def load_test_dataset(dir: str):
    """Convert the CIFAR10 test set into a tensor dataset."""
    # Root download the data if not already available.
    test_data_raw = torchvision.datasets.CIFAR10(
        dir, download=True, train=False, transform=transforms.ToTensor()
    )

    # Obtain the testdataset
    test_data_input = []
    test_data_label = []
    for idx in range(len(test_data_raw)):
        test_data_input.append(test_data_raw[idx][0].tolist())
        test_data_label.append(test_data_raw[idx][1])
    return Dataset(torch.FloatTensor(test_data_input), torch.tensor(test_data_label))


def partition_train_dataset(
    dir: str, num_clients: int, partition_strategy: str, **kwargs
):
    """Split the CIFAR10 training set into one tensor dataset per client."""
    # Training data for multiple clients
    train_data_raw = torchvision.datasets.CIFAR10(
        dir, download=False, train=True, transform=transforms.ToTensor()
    )

    # Partition the dataset
    if partition_strategy == "iid":
        return iid_partition(train_data_raw, num_clients)
    elif partition_strategy == "class_noniid":
        return class_noniid_partition(train_data_raw, num_clients, **kwargs)
    elif partition_strategy == "dirichlet_noniid":
        return dirichlet_noniid_partition(train_data_raw, num_clients, **kwargs)
    else:
        raise ValueError(f"Invalid partition strategy: {partition_strategy}")


def get_cifar10(
    num_clients: int, client_id: int, partition_strategy: str = "iid", **kwargs
):
    """
    Return the CIFAR10 dataset for a given client.

    The test set and the partition are built once per process and shared by all
    the clients created in it.
    :param num_clients: total number of clients
    :param client_id: the client id
    """
    # Get the download directory for dataset
    dir = os.getcwd() + "/datasets/RawData"

    test_dataset = cache_per_process("cifar10_dataset.test", load_test_dataset, dir=dir)
    train_datasets = cache_per_process(
        "cifar10_dataset.train",
        partition_train_dataset,
        dir=dir,
        num_clients=num_clients,
        partition_strategy=partition_strategy,
        **kwargs,
    )
    return train_datasets[client_id], test_dataset
