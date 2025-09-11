import os
import torch
import torchvision
import torchvision.transforms as transforms
from appfl.misc.data import (
    Dataset,
    iid_partition,
    class_noniid_partition,
    dirichlet_noniid_partition,
)


def get_custom_mnist(
    num_clients: int,
    client_id: int,
    partition_strategy: str = "dirichlet_noniid",
    **kwargs,
):
    """
    Return the MNIST dataset for a given client with custom partitioning.
    :param num_clients: total number of clients
    :param client_id: the client id
    :param partition_strategy: data partitioning strategy
    """
    # Read reproducibility seed if provided
    seed = kwargs.get("seed", 42)

    # Get the download directory for dataset
    dir = os.getcwd() + "/datasets/RawData"

    # Ensure directory exists
    os.makedirs(dir, exist_ok=True)

    # Download the test data if not already available
    test_data_raw = torchvision.datasets.MNIST(
        dir, download=True, train=False, transform=transforms.ToTensor()
    )

    # Obtain the test dataset
    test_data_input = []
    test_data_label = []
    for idx in range(len(test_data_raw)):
        test_data_input.append(test_data_raw[idx][0].tolist())
        test_data_label.append(test_data_raw[idx][1])
    test_dataset = Dataset(
        torch.FloatTensor(test_data_input), torch.tensor(test_data_label)
    )

    # Training data for multiple clients
    train_data_raw = torchvision.datasets.MNIST(
        dir, download=False, train=True, transform=transforms.ToTensor()
    )

    # Partition the dataset based on strategy
    if partition_strategy == "iid":
        train_datasets = iid_partition(train_data_raw, num_clients)
    elif partition_strategy == "class_noniid":
        # Use class-based non-IID partitioning
        train_datasets = class_noniid_partition(
            train_data_raw,
            num_clients,
            classes_per_client=kwargs.get("classes_per_client", 2),
            seed=seed,
        )
    elif partition_strategy == "dirichlet_noniid":
        # Use Dirichlet distribution for non-IID partitioning
        # Note: dirichlet_noniid_partition uses alpha1 and alpha2; legacy 'alpha' is ignored there.
        # We pass the seed to ensure deterministic partitions across runs.
        train_datasets = dirichlet_noniid_partition(
            train_data_raw,
            num_clients,
            seed=seed,
        )
    else:
        raise ValueError(f"Invalid partition strategy: {partition_strategy}")

    # Handle visualization if requested
    if kwargs.get("visualization", False) and client_id == 0:
        print(f"Dataset partitioned using {partition_strategy}")
        print(f"Number of clients: {num_clients}")
        if partition_strategy == "dirichlet_noniid":
            print(f"Dirichlet alpha: {kwargs.get('alpha', 0.1)}")

        # Print client data distribution
        client_dataset = train_datasets[client_id]
        if hasattr(client_dataset, "targets"):
            targets = client_dataset.targets
        else:
            targets = [client_dataset[i][1] for i in range(len(client_dataset))]

        unique, counts = torch.unique(torch.tensor(targets), return_counts=True)
        print(
            f"Client {client_id} class distribution: {dict(zip(unique.tolist(), counts.tolist()))}"
        )

    return train_datasets[client_id], test_dataset
