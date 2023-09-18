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

def dirichlet_noiid(comm: MPI.Comm, cfg: DictConfig, dataset: str, alpha1: float = 8, alpha2: float = 0.5, seed: int = 42, visualization: bool = True, **kwargs):
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

    classes = [len(label_indices[label]) for label in labels]

    normalized_portions = np.zeros(individuals.shape)
    for i in range(num_clients):
        for j in range(len(classes)):
            normalized_portions[i][j] = weights[i] * individuals[i][j] / np.dot(weights, individuals.transpose()[j])

    res = np.multiply(np.array([classes] * num_clients), normalized_portions).transpose()

    for i in range(len(classes)):
        total = 0
        for j in range(num_clients - 1):
            res[i][j] = int(res[i][j])
            total += res[i][j]
        res[i][num_clients - 1] = classes[i] - total

    if comm_rank == 0 and visualization:
        dir = cfg.output_dirname
        if os.path.isdir(dir) == False:
            os.makedirs(dir, exist_ok=True)
        output_filename = f"{dataset}_{num_clients}clients_dirichlet_distribution_{seed}"
        file_ext = ".pdf"
        filename = dir + "/%s%s" % (output_filename, file_ext)
        uniq = 1
        while os.path.exists(filename):
            filename = dir + "/%s_%d%s" % (output_filename, uniq, file_ext)
            uniq += 1
        plot(num_clients, classes, res, filename)

    # number of elements from each class for each client
    num_elements = np.array(res.transpose(), dtype=np.int32)
    sum_elements = np.cumsum(num_elements, axis=0)

    train_datasets = []
    for i in range(num_clients):
        train_data_input = []
        train_data_label = []
        for j, label in enumerate(labels):
            start = 0 if i == 0 else sum_elements[i-1][j]
            end = sum_elements[i][j]
            for idx in label_indices[label][start:end]:
                train_data_input.append(train_data_raw[idx][0].tolist())
                train_data_label.append(train_data_raw[idx][1])
        train_datasets.append(
            Dataset(
                torch.FloatTensor(train_data_input),
                torch.tensor(train_data_label),
            )
        )

    return train_datasets, test_dataset
