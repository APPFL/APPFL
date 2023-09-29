import os
import torch
import torchvision
import numpy as np
from .utils import *
from mpi4py import MPI
from typing import Optional
from appfl.config import *
from appfl.misc.data import *
from appfl.misc.utils import *
from omegaconf import DictConfig

def iid(comm: Optional[MPI.Comm], cfg: DictConfig, dataset: str, visualization: bool = True, **kwargs):
    comm_rank = comm.Get_rank() if comm is not None else 0
    num_clients = cfg.num_clients
    
    # Get the download directory for dataset
    dir = os.getcwd() + "/datasets/RawData"

    # Root download the data if not already available.
    if comm_rank == 0:
        test_data_raw = eval("torchvision.datasets." + dataset)(dir, download=True, train=False, transform=test_transform(dataset))
    if comm is not None:
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

    label_counts = {}
    client_dataset_info = {}

    # training data for multiple clients
    train_data_raw = eval("torchvision.datasets." + dataset)(dir, download=False, train=True, transform=train_transform(dataset))

    split_train_data_raw = np.array_split(range(len(train_data_raw)), num_clients)
    train_datasets = []
    for i in range(num_clients):
        client_dataset_info[i] = {}
        train_data_input = []
        train_data_label = []
        for idx in split_train_data_raw[i]:
            train_data_input.append(train_data_raw[idx][0].tolist())
            train_data_label.append(train_data_raw[idx][1])
            label = train_data_raw[idx][1]
            if not label in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
            if not label in client_dataset_info[i]:
                client_dataset_info[i][label] = 0
            client_dataset_info[i][label] += 1
        train_datasets.append(
            Dataset(
                torch.FloatTensor(train_data_input),
                torch.tensor(train_data_label),
            )
        )

    # Visualize the data distribution among clients
    if comm_rank == 0 and visualization:
        dir = cfg.output_dirname
        if os.path.isdir(dir) == False:
            os.mkdir(dir)
        output_filename = f"{dataset}_{num_clients}clients_iid_distribution"
        file_ext = ".pdf"
        filename = dir + "/%s%s" % (output_filename, file_ext)
        uniq = 1
        while os.path.exists(filename):
            filename = dir + "/%s_%d%s" % (output_filename, uniq, file_ext)
            uniq += 1
        classes = []
        for i in range(len(label_counts)):
            classes.append(label_counts[i])
        res = np.zeros((len(classes), num_clients))
        for i in range(num_clients):
            for cls in client_dataset_info[i]:
                res[cls][i] = client_dataset_info[i][cls]
        plot(num_clients, classes, res, filename)

    return train_datasets, test_dataset
