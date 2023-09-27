import os
import torch
import torchvision
import numpy as np
from .utils import *
from mpi4py import MPI
from appfl.config import *
from appfl.misc.data import *
from appfl.misc.utils import *
from omegaconf import DictConfig

def class_noiid(comm: MPI.Comm, cfg: DictConfig, dataset: str, seed: int = 42, visualization: bool = True, **kwargs):
    comm_rank = comm.Get_rank()
    num_clients = cfg.num_clients

    # Set a random seed for same dataset partition among all clients
    np.random.seed(seed)

    # Get the download directory for dataset
    dir = os.getcwd() + "/datasets/RawData"

    # Root download the data if not already available.
    if comm_rank == 0:
        test_data_raw = eval("torchvision.datasets." + dataset)(dir, download=True, train=False, transform=test_transform(dataset))
    comm.Barrier()
    if comm_rank > 0:
        test_data_raw = eval("torchvision.datasets." + dataset)(dir, download=False, train=False, transform=test_transform(dataset))

    # Obtain the testdataset
    test_data_input = []
    test_data_label = []
    for idx in range(len(test_data_raw)):
        test_data_input.append(test_data_raw[idx][0].tolist())
        test_data_label.append(test_data_raw[idx][1])
    test_dataset = Dataset(torch.FloatTensor(test_data_input), torch.tensor(test_data_label))

    # training data for multiple clients
    Cmin = {1: 10, 2: 7, 3: 6, 4: 5, 5: 5, 6: 4, 7: 4, 'none': 3}       # minimum sample classes for each client
    Cmax = {1: 10, 2: 8, 3: 8, 4: 7, 5: 6, 6: 6, 7: 5, 'none': 5}       # maximum sample classes for each client
    train_data_raw = eval("torchvision.datasets." + dataset)(dir, download=False, train=True, transform=train_transform(dataset))

    # Split the dataset by label
    labels = []
    label_indices = {}
    for idx, (_, label) in enumerate(train_data_raw):
        if label not in label_indices:
            label_indices[label] = []
            labels.append(label)
        label_indices[label].append(idx)
    labels.sort()

    # Obtain the way to partition the dataset
    while True:
        class_partition = {}    # number of partitions for each class of MNIST
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
    if comm_rank == 0 and visualization:
        dir = cfg.output_dirname
        if os.path.isdir(dir) == False:
            os.mkdir(dir)
        output_filename = f"{dataset}_{num_clients}clients_partition_distribution_{seed}"
        file_ext = ".pdf"
        filename = dir + "/%s%s" % (output_filename, file_ext)
        uniq = 1
        while os.path.exists(filename):
            filename = dir + "/%s_%d%s" % (output_filename, uniq, file_ext)
            uniq += 1
        classes = [len(label_indices[label]) for label in labels]
        res = np.zeros((len(classes), num_clients))
        for i in range(num_clients):
            for cls in client_dataset_info[i]:
                res[cls][i] = client_dataset_info[i][cls]
        plot(num_clients, classes, res, filename)

    train_datasets = []
    for i in range(num_clients):
        train_data_input = []
        train_data_label = []
        for idx in client_datasets[i]:
            train_data_input.append(train_data_raw[idx][0].tolist())
            train_data_label.append(train_data_raw[idx][1])
        train_datasets.append(
            Dataset(
                torch.FloatTensor(train_data_input),
                torch.tensor(train_data_label),
            )
        )
    return train_datasets, test_dataset
