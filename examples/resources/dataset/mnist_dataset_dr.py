import os
import torch
import torchvision
from omegaconf import OmegaConf
import torchvision.transforms as transforms
from appfl.misc.data import (
    Dataset,
    cache_per_process,
    iid_partition,
    class_noniid_partition_binary,
    dirichlet_noniid_partition,
)
from appfl.misc.data_readiness.data_pollute import add_noise_to_subset, add_duplicates


def load_test_dataset(dir: str):
    """Convert the MNIST test set into a binary (digit < 5 or not) tensor dataset."""
    # Root download the data if not already available.
    test_data_raw = torchvision.datasets.MNIST(
        dir, download=True, train=False, transform=transforms.ToTensor()
    )

    # convert to binary classification task
    test_data_binary = [(img, 0 if label < 5 else 1) for img, label in test_data_raw]
    test_data_raw = Dataset(*zip(*test_data_binary))

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
    """Split the binary MNIST training set into one tensor dataset per client."""
    # Training data for multiple clients
    train_data_raw = torchvision.datasets.MNIST(
        dir, download=False, train=True, transform=transforms.ToTensor()
    )

    # Convert to binary classification task
    train_data_binary = [(img, 0 if label < 5 else 1) for img, label in train_data_raw]
    train_data_raw = Dataset(*zip(*train_data_binary))

    if partition_strategy == "iid":
        return iid_partition(train_data_raw, num_clients)
    elif partition_strategy == "class_noniid":
        return class_noniid_partition_binary(train_data_raw, num_clients, **kwargs)
    elif partition_strategy == "dirichlet_noniid":
        return dirichlet_noniid_partition(train_data_raw, num_clients, **kwargs)

    else:
        raise ValueError(f"Invalid partition strategy: {partition_strategy}")


def get_mnist(
    num_clients: int, client_id: int, partition_strategy: str = "iid", **kwargs
):
    """
    Return the MNIST dataset for a given client.

    The test set and the partition are built once per process and shared by all
    the clients created in it.
    :param num_clients: total number of clients
    :param client_id: the client id
    """
    # Get the download directory for dataset
    dir = os.getcwd() + "/datasets/RawData"
    server_agent_config = OmegaConf.load(
        "./resources/configs/mnist_dr/server_fedavg_cadremodule.yaml"
    )

    # Load the server agent config to get the pollution method based on the CADFEModule name
    if hasattr(server_agent_config, "data_readiness_configs") and hasattr(
        server_agent_config.client_configs.data_readiness_configs.dr_metrics.cadre_module_configs,
        "cadremodule_name",
    ):
        pollution_method = server_agent_config.client_configs.data_readiness_configs.dr_metrics.cadremodule_configs.cadremodule_name
    else:
        pollution_method = None

    test_dataset = cache_per_process(
        "mnist_dataset_dr.test", load_test_dataset, dir=dir
    )
    train_datasets = cache_per_process(
        "mnist_dataset_dr.train",
        partition_train_dataset,
        dir=dir,
        num_clients=num_clients,
        partition_strategy=partition_strategy,
        **kwargs,
    )

    # adding different pollution techniques to clients; the polluted copy stays
    # local, so the shared partition is left clean
    train_dataset = train_datasets[client_id]
    if (
        pollution_method == "CADREModuleNoise"
        or pollution_method == "CADREModuleOutliers"
    ):
        # adding noise to the dataset
        train_dataset = add_noise_to_subset(train_dataset, scale=2, fraction=0.20)
    elif (
        pollution_method == "CADREModuleDuplicates"
        or pollution_method == "CADREModuleMemory"
    ):
        # adding duplicates to the dataset
        train_dataset = add_duplicates(train_dataset, duplicate_ratio=0.5)
    else:
        # No pollution method specified, use the original dataset
        pass
    return train_dataset, test_dataset
