"""
Miscellaneous data classes and processing functions for federated learning.
"""

import os
import torch
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from appfl.config import *
from torch.utils import data
from typing import List, Optional
from .deprecation import deprecated

class Dataset(data.Dataset):
    """
    This class provides a simple way to define client dataset for supervised learning.
    This is derived from ``torch.utils.data.Dataset`` so that can be loaded to ``torch.utils.data.DataLoader``.
    Users may also create their own dataset class derived from this for more data processing steps.

    An empty ``Dataset`` class is created if no argument is given (i.e., ``Dataset()``).

    :param data_input (`torch.FloatTensor`): optional data inputs
    :param data_label (`torch.Tensor`): optional data ouputs (or labels)
    """

    def __init__(
        self,
        data_input: torch.FloatTensor = torch.FloatTensor(),
        data_label: torch.Tensor = torch.Tensor(),
    ):
        self.data_input = data_input
        self.data_label = data_label

    def __len__(self):
        """This returns the sample size."""
        return len(self.data_label)

    def __getitem__(self, idx):
        """This returns a sample point for given ``idx``."""
        return self.data_input[idx], self.data_label[idx]

def plot_distribution(
    num_clients: int, 
    classes_samples: List[int], 
    sample_matrix: np.ndarray, 
    output_dirname: Optional[str],
    output_filename: Optional[str]
):
    """
    Visualize the data distribution among clients for different classes.
    :param num_clients: number of clients
    :param classes_samples: number of samples for each class
    :param sample_matrix: the number of samples for each class for each client with shape (num_classes, num_clients)
    :param file_name: the filename to save the plot
    """
    _, ax = plt.subplots(figsize=(20, num_clients/2+3))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    colors = [
        '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', 
        '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5', 
        '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', 
        '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5'
    ]

    for i in range(len(classes_samples)):
        ax.barh(
            y=range(num_clients),
            width=sample_matrix[i],
            left=np.sum(sample_matrix[:i], axis=0) if i > 0 else 0,
            color=colors[i],
        )

    ax.set_ylabel("Client")
    ax.set_xlabel("Number of Elements")
    ax.set_xticks([])
    ax.set_yticks([])
    output_dirname = "output" if output_dirname is None else output_dirname
    output_filename = "data_distribution.pdf" if output_filename is None else output_filename
    output_filename = f"{output_filename}.pdf" if not output_filename.endswith(".pdf") else output_filename
    if not os.path.exists(output_dirname):
        pathlib.Path(output_dirname).mkdir(parents=True, exist_ok=True)
    unique = 1
    unique_filename = output_filename
    filename_base, ext = os.path.splitext(output_filename)
    while pathlib.Path(os.path.join(output_dirname, unique_filename)).exists():
        unique_filename = f"{filename_base}_{unique}{ext}"
        unique += 1
    plt.savefig(os.path.join(output_dirname, unique_filename))

def iid_partition(
    train_dataset: data.Dataset, 
    num_clients: int,
) -> List[data.Dataset]:
    """
    Partition a `torch.utils.data.Dataset` into `num_clients` clients chunks in an IID manner.
    :param train_dataset: the training dataset
    :param num_clients: number of clients
    :return train_dataset_partitioned: a list of `torch.utils.data.Dataset` for each client
    """
    train_dataset_split_indices = np.array_split(range(len(train_dataset)), num_clients)
    train_dataset_partitioned = []
    for i in range(num_clients):
        train_data_input = []
        train_data_label = []
        for idx in train_dataset_split_indices[i]:
            train_data_input.append(train_dataset[idx][0].tolist())
            train_data_label.append(train_dataset[idx][1])
        train_dataset_partitioned.append(
            Dataset(
                torch.FloatTensor(train_data_input),
                torch.tensor(train_data_label),
            )
        )
    return train_dataset_partitioned

def class_noniid_partition(
    train_dataset: data.Dataset, 
    num_clients: int, 
    visualization: bool=False, 
    output_dirname: Optional[str]=None, 
    output_filename: Optional[str]=None,
    seed: int=42, 
    **kwargs
):
    """
    Partition a `torch.utils.data.Dataset` into `num_clients` clients chunks in a 
    non-IID manner by letting each client only have a subset of all classes.
    :param train_dataset: the training dataset
    :param num_clients: number of clients
    :param visualization: whether to visualize the data distribution among clients
    :param output_dirname: the directory to save the plot
    :param output_filename: the filename to save the plot
    :param seed: the random seed
    :return train_dataset_partitioned: a list of `torch.utils.data.Dataset` for each client
    """
    np.random.seed(seed)
    # training data for multiple clients
    Cmin = {1: 10, 2: 7, 3: 6, 4: 5, 5: 5, 6: 4, 7: 4, 'none': 3}       # minimum sample classes for each client
    Cmax = {1: 10, 2: 8, 3: 8, 4: 7, 5: 6, 6: 6, 7: 5, 'none': 5}       # maximum sample classes for each client

    # Split the dataset by label
    labels = []
    label_indices = {}
    for idx, (_, label) in enumerate(train_dataset):
        if label not in label_indices:
            label_indices[label] = []
            labels.append(label)
        label_indices[label].append(idx)
    labels.sort()

    # Obtain the way to partition the dataset
    while True:
        class_partition = {}    # number of partitions for each class of CIFAR10
        client_classes  = {}    # sample classes for different clients
        for i in range(num_clients):
            cmin = Cmin[num_clients] if num_clients in Cmin else Cmin['none']
            cmax = Cmax[num_clients] if num_clients in Cmax else Cmax['none']
            cnum = np.random.randint(cmin, cmax+1)
            classes = np.random.permutation(range(10))[:cnum]
            client_classes[i] = classes 
            for cls in classes: 
                if cls in class_partition:
                    class_partition[cls] += 1
                else:
                    class_partition[cls] = 1
        if len(class_partition) == 10: break
            
    # Calculate how to partition the dataset
    partition_endpoints = {}
    for label in labels:
        total_size = len(label_indices[label])

        # Partiton the samples from the same class to different lengths
        partitions = class_partition[label]
        partition_lengths = np.abs(np.random.normal(10, 3, size=partitions))

        # Scale the lengths so they add to the total length
        partition_lengths = partition_lengths / np.sum(partition_lengths) * total_size

        # Calculate the endpoints of each subrange
        endpoints = np.cumsum(partition_lengths)
        endpoints = np.array(endpoints, dtype=np.int32)
        endpoints[-1] = total_size
        partition_endpoints[label] = endpoints
    
    # Start dataset partition
    partition_pointer = {}
    for label in labels:
        partition_pointer[label] = 0
    client_datasets = []
    client_dataset_info = {}
    for i in range(num_clients):
        client_dataset_info[i] = {}
        sample_indices = []
        client_class = client_classes[i]
        for cls in client_class:
            start_idx = 0 if partition_pointer[cls] == 0 else partition_endpoints[cls][partition_pointer[cls]-1] # included
            end_idx = partition_endpoints[cls][partition_pointer[cls]] # excluded
            sample_indices.extend(label_indices[cls][start_idx:end_idx])
            partition_pointer[cls] += 1
            client_dataset_info[i][cls] = end_idx - start_idx # record the number for different classes
        client_datasets.append(sample_indices)

    # Visualize the data distirbution among clients
    if visualization:
        classes_samples = [len(label_indices[label]) for label in labels]
        sample_matrix = np.zeros((len(classes_samples), num_clients))
        for i in range(num_clients):
            for cls in client_dataset_info[i]:
                sample_matrix[cls][i] = client_dataset_info[i][cls]
        plot_distribution(num_clients, classes_samples, sample_matrix, output_dirname, output_filename)

    train_datasets = []
    for i in range(num_clients):
        train_data_input = []
        train_data_label = []
        for idx in client_datasets[i]:
            train_data_input.append(train_dataset[idx][0].tolist())
            train_data_label.append(train_dataset[idx][1])

        train_datasets.append(
            Dataset(
                torch.FloatTensor(train_data_input),
                torch.tensor(train_data_label),
            )
        )
    return train_datasets

def dirichlet_noniid_partition(
    train_dataset: data.Dataset, 
    num_clients: int, 
    visualization: bool=False, 
    output_dirname: Optional[str]=None,
    output_filename: Optional[str]=None,
    alpha1: int=8, 
    alpha2: int=0.5, 
    seed: int=42, 
    **kwargs
):
    """
    Partition a `torch.utils.data.Dataset` into `num_clients` clients chunks
    using two Dirichlet distributions: one for the number of elements for each client
    and the other for the number of elements from each class for each client.
    :param train_dataset: the training dataset
    :param num_clients: number of clients
    :param visualization: whether to visualize the data distribution among clients
    :param output_dirname: the directory to save the plot
    :param output_filename: the filename to save the plot
    :param alpha1: the concentration parameter for the Dirichlet distribution for the number of elements for each client
    :param alpha2: the concentration parameter for the Dirichlet distribution for the number of elements from each class for each client
    :param seed: the random seed
    """
    np.random.seed(seed)
    # Split the dataset by label
    labels = []
    label_indices = {}
    for idx, (_, label) in enumerate(train_dataset):
        if label not in label_indices:
            label_indices[label] = []
            labels.append(label)
        label_indices[label].append(idx)
    labels.sort()

    # Shuffle the indices for different label
    for label in labels:
        np.random.shuffle(label_indices[label])

    p1 = [1 / num_clients for _ in range(num_clients)]      # prior distribution for each client's number of elements
    p2 = [len(label_indices[label]) for label in labels]
    p2 = [p / sum(p2) for p in p2]                          # prior distribution for each class's number of elements

    q1 = [alpha1 * i for i in p1]
    q2 = [alpha2 * i for i in p2]

    weights = np.random.dirichlet(q1) # the total number of elements for each client
    individuals = np.random.dirichlet(q2, num_clients) # the number of elements from each class for each client

    classes_samples = [len(label_indices[label]) for label in labels]

    normalized_portions = np.zeros(individuals.shape)
    for i in range(num_clients):
        for j in range(len(classes_samples)):
            normalized_portions[i][j] = weights[i] * individuals[i][j] / np.dot(weights, individuals.transpose()[j])

    sample_matrix = np.multiply(np.array([classes_samples] * num_clients), normalized_portions).transpose()

    for i in range(len(classes_samples)):
        total = 0
        for j in range(num_clients - 1):
            sample_matrix[i][j] = int(sample_matrix[i][j])
            total += sample_matrix[i][j]
        sample_matrix[i][num_clients - 1] = classes_samples[i] - total

    if visualization:
        plot_distribution(num_clients, classes_samples, sample_matrix, output_dirname, output_filename)

    # number of elements from each class for each client
    num_elements = np.array(sample_matrix.transpose(), dtype=np.int32)
    sum_elements = np.cumsum(num_elements, axis=0)

    train_datasets = []
    for i in range(num_clients):
        train_data_input = []
        train_data_label = []
        for j, label in enumerate(labels):
            start = 0 if i == 0 else sum_elements[i-1][j]
            end = sum_elements[i][j]
            for idx in label_indices[label][start:end]:
                train_data_input.append(train_dataset[idx][0].tolist())
                train_data_label.append(train_dataset[idx][1])
        train_datasets.append(
            Dataset(
                torch.FloatTensor(train_data_input),
                torch.tensor(train_data_label),
            )
        )
    return train_datasets

@deprecated(silent=True)
def data_sanity_check(train_datasets, test_dataset, num_channel, num_pixel):

    ## Check if "DataLoader" from PyTorch works.
    train_dataloader = data.DataLoader(train_datasets[0], batch_size=64, shuffle=False)

    for input, label in train_dataloader:

        assert input.shape[0] == label.shape[0]
        assert input.shape[1] == num_channel
        assert input.shape[2] == num_pixel
        assert input.shape[3] == num_pixel

    test_dataloader = data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    for input, label in test_dataloader:

        assert input.shape[0] == label.shape[0]
        assert input.shape[1] == num_channel
        assert input.shape[2] == num_pixel
        assert input.shape[3] == num_pixel
