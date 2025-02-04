import os
import torch
import torchvision
import torchvision.transforms as transforms
from appfl.misc.data import Dataset, iid_partition, class_noniid_partition, dirichlet_noniid_partition
from appfl.misc.data_readiness import *
from omegaconf import OmegaConf



def get_mnist(
    num_clients: int,
    client_id: int,
    partition_strategy: str = "iid",
    **kwargs
):
    """
    Return the MNIST dataset for a given client.
    :param num_clients: total number of clients
    :param client_id: the client id
    """

    client_agent_config = OmegaConf.load("./resources/configs/mnist/client_1.yaml")
    noise_prop = client_agent_config.data_configs.dataset_kwargs.noise_prop
    sample_size = client_agent_config.data_configs.dataset_kwargs.sample_size

    # Get the download directory for dataset
    dir = os.getcwd() + "/datasets/RawData"

    # Root download the data if not already available.
    test_data_raw = torchvision.datasets.MNIST(dir, download=True, train=False, transform=transforms.ToTensor())
    test_data_binary = [(img, 0 if label < 5 else 1) for img, label in test_data_raw]
    test_data_raw = Dataset(*zip(*test_data_binary))

    # Obtain the testdataset
    test_data_input = []
    test_data_label = []
    for idx in range(len(test_data_raw)):
        test_data_input.append(test_data_raw[idx][0].tolist())
        test_data_label.append(test_data_raw[idx][1])
    test_dataset = Dataset(torch.FloatTensor(test_data_input), torch.tensor(test_data_label))

    # Training data for multiple clients
    train_data_raw = torchvision.datasets.MNIST(dir, download=False, train=True, transform=transforms.ToTensor())
    train_data_binary = [(img, 0 if label < 5 else 1) for img, label in train_data_raw]
    train_data_raw = Dataset(*zip(*train_data_binary))

    # Partition the dataset
    if partition_strategy == "iid":
        train_datasets = iid_partition(train_data_raw, num_clients)
    elif partition_strategy == "class_noniid":
        train_datasets = class_noniid_partition(train_data_raw, num_clients, **kwargs)
    elif partition_strategy == "dirichlet_noniid":
        train_datasets = dirichlet_noniid_partition(train_data_raw, num_clients, **kwargs)
    else:
        raise ValueError(f"Invalid partition strategy: {partition_strategy}")

    # perct_client_ids = 0.9

    # if client_id < int(perct_client_ids * num_clients):
    # if client_id == 0:
    #     train_datasets[client_id] = add_noise_to_subset(train_datasets[client_id], scale=2, fraction=0.95)
    # elif client_id == 1:
    #     train_datasets[client_id] = add_noise_to_subset(train_datasets[client_id], scale=2, fraction=0.95)
    
    if noise_prop >= 0:
        train_datasets[client_id] = add_noise_to_subset(train_datasets[client_id], scale=2, fraction=noise_prop)
    
    if sample_size >= 0:
        train_datasets[client_id] = sample_subset(train_datasets[client_id], sample_size)

    print(len(test_dataset))

    return train_datasets[client_id], test_dataset


    
