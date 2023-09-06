import pytest
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import ToTensor

import math
import numpy as np

from appfl.config import *
from appfl.misc.data import *
import appfl.run_mpi as rm
import appfl.run_serial as rs
import appfl.run_mpi_async as rma


from mpi4py import MPI
import os


class CNN(nn.Module):
    def __init__(self, num_channel, num_classes, num_pixel):
        super().__init__()
        self.conv1 = nn.Conv2d(
            num_channel, 32, kernel_size=5, padding=0, stride=1, bias=True
        )
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        self.act = nn.ReLU(inplace=True)

        X = num_pixel
        X = math.floor(1 + (X + 2 * 0 - 1 * (5 - 1) - 1) / 1)
        X = X / 2
        X = math.floor(1 + (X + 2 * 0 - 1 * (5 - 1) - 1) / 1)
        X = X / 2
        X = int(X)

        self.fc1 = nn.Linear(64 * X * X, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.maxpool(x)
        x = self.act(self.conv2(x))
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x

def process_data(num_clients):
    # test data for a server
    test_data_raw = torchvision.datasets.MNIST(
        "./_data", download=False, train=False, transform=ToTensor()
    )

    test_data_input = []
    test_data_label = []
    for idx in range(len(test_data_raw)):
        test_data_input.append(test_data_raw[idx][0].tolist())
        test_data_label.append(test_data_raw[idx][1])

    test_dataset = Dataset(
        torch.FloatTensor(test_data_input), torch.tensor(test_data_label)
    )

    # training data for multiple clients
    train_data_raw = torchvision.datasets.MNIST(
        "./_data", download=False, train=True, transform=ToTensor()
    )

    split_train_data_raw = np.array_split(range(len(train_data_raw)), num_clients)
    train_datasets = []
    for i in range(num_clients):
        train_data_input = []
        train_data_label = []
        for idx in split_train_data_raw[i]:
            train_data_input.append(train_data_raw[idx][0].tolist())
            train_data_label.append(train_data_raw[idx][1])

        train_datasets.append(
            Dataset(
                torch.FloatTensor(train_data_input),
                torch.tensor(train_data_label),
            )
        )

    return train_datasets, test_dataset


# Let's download the data first if data does not exist.
def readyMNISTdata():    
    currentpath = os.getcwd()    
    datafolderpath = os.path.join(currentpath, "_data")
    
    if not (os.path.exists(datafolderpath) and os.path.isdir(datafolderpath)):
        os.mkdir(datafolderpath)

    mnistfolderpath = os.path.join(datafolderpath, "MNIST")
    if not (os.path.exists(mnistfolderpath) and os.path.isdir(mnistfolderpath)):        
        print("Download MNIST data")
        torchvision.datasets.MNIST(
            "./_data", download=True, train=False, transform=ToTensor()
        )

comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
if comm_size > 1:
    comm_rank = comm.Get_rank()
    if comm_rank == 0:
        readyMNISTdata()
    comm.Barrier()
else:
    # Serial
    readyMNISTdata() 

def test_mnist_fedavg():

    num_clients = 2
    cfg = OmegaConf.structured(Config)
    cfg.fed.args.num_local_epochs=2 

    model = CNN(1, 10, 28)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_datasets, test_dataset = process_data(num_clients)

    rs.run_serial(cfg, model, loss_fn, train_datasets, test_dataset, "test_mnist")


@pytest.mark.mpi(min_size=2)
def test_mnist_fedavg_mpi(): 

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    num_clients = 2
    cfg = OmegaConf.structured(Config)
    cfg.fed.args.num_local_epochs=2
    model = CNN(1, 10, 28)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_datasets, test_dataset = process_data(num_clients)

    if comm_size > 1:
        if comm_rank == 0:
            rm.run_server(cfg, comm, model, loss_fn, num_clients, test_dataset, "test_mnist")
        else:
            rm.run_client(cfg, comm, model, loss_fn, num_clients, train_datasets)
    else:
        assert 0


@pytest.mark.mpi(min_size=2)
def test_mnist_iceadmm_mpi():
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    num_clients = 2
    cfg = OmegaConf.structured(Config(fed=ICEADMM()))
    cfg.fed.args.num_local_epochs=2
    model = CNN(1, 10, 28)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_datasets, test_dataset = process_data(num_clients)

    if comm_size > 1:
        if comm_rank == 0:
            rm.run_server(cfg, comm, model, loss_fn, num_clients, test_dataset, "test_mnist")
        else:
            rm.run_client(cfg, comm, model, loss_fn, num_clients, train_datasets)
    else:
        assert 0


@pytest.mark.mpi(min_size=2)
def test_mnist_iiadmm_mpi():
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    num_clients = 2
    cfg = OmegaConf.structured(Config(fed=IIADMM()))
    cfg.fed.args.num_local_epochs=2
    model = CNN(1, 10, 28)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_datasets, test_dataset = process_data(num_clients)

    if comm_size > 1:
        if comm_rank == 0:
            rm.run_server(cfg, comm, model, loss_fn, num_clients, test_dataset, "test_mnist")
        else:
            rm.run_client(cfg, comm, model, loss_fn, num_clients, train_datasets)
    else:
        assert 0


@pytest.mark.mpi(min_size=2)
def test_mnist_fedasync_mpi(): 
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    num_clients = comm_size - 1
    cfg = OmegaConf.structured(Config(fed=FedAsync()))
    cfg.fed.args.num_local_epochs=2
    cfg.fed.args.staleness_func.name = 'polynomial'
    model = CNN(1, 10, 28)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_datasets, test_dataset = process_data(num_clients)
    
    if comm_size > 1:
        if comm_rank == 0:
            rma.run_server(cfg, comm, model, loss_fn, num_clients, test_dataset, "test_mnist")
        else:
            rma.run_client(cfg, comm, model, loss_fn, num_clients, train_datasets)
    else:
        assert 0


@pytest.mark.mpi(min_size=2)
def test_mnist_fedbuffer_mpi(): 
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    num_clients = comm_size - 1
    cfg = OmegaConf.structured(Config(fed=FedAsync()))
    cfg.num_epochs = 3
    cfg.fed.args.num_local_epochs=2
    cfg.fed.args.gradient_based = True
    cfg.fed.args.staleness_func.name = 'polynomial'
    cfg.fed.servername = 'ServerFedBuffer'
    cfg.fed.args.K = 2

    model = CNN(1, 10, 28)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_datasets, test_dataset = process_data(num_clients)
    
    if comm_size > 1:
        if comm_rank == 0:
            rma.run_server(cfg, comm, model, loss_fn, num_clients, test_dataset, "test_mnist")
        else:
            rma.run_client(cfg, comm, model, loss_fn, num_clients, train_datasets)
    else:
        assert 0


def test_mnist_fedavg_notest():
    num_clients = 2
    cfg = OmegaConf.structured(Config)
    cfg.fed.args.num_local_epochs=2
    model = CNN(1, 10, 28)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_datasets, test_dataset = process_data(num_clients)

    rs.run_serial(cfg, model, loss_fn, train_datasets, Dataset(), "test_mnist")

@pytest.mark.mpi(min_size=2)
def test_mnist_fedavg_mpi_notest():
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    num_clients = 2
    cfg = OmegaConf.structured(Config)
    cfg.fed.args.num_local_epochs=2
    model = CNN(1, 10, 28)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_datasets, test_dataset = process_data(num_clients)

    if comm_size > 1:
        if comm_rank == 0:
            rm.run_server(cfg, comm, model, loss_fn, num_clients, Dataset(), "test_mnist")
        else:
            rm.run_client(cfg, comm, model, loss_fn, num_clients, train_datasets)
    else:
        assert 0


# mpirun -n 3 python -m pytest --with-mpi
