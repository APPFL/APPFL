
import os
from collections import OrderedDict
import torch.nn as nn
from torch.optim import *
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler
import torchvision
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from mpi4py import MPI

import hydra
from omegaconf import DictConfig

import copy
import time
from .algorithm.iadmm import *
from .algorithm.fedavg import *


def run_serial(cfg: DictConfig, model: nn.Module, train_data, test_data=None):

    num_clients = cfg.num_clients
    num_epochs = cfg.num_epochs

    local_data_size = int(len(train_data) / num_clients)
    how_to_split = [local_data_size for i in range(num_clients)]
    how_to_split[-1] += len(train_data) - sum(how_to_split)
    datasets = data.random_split(train_data, how_to_split)

    # print(cfg.model.classname)
    optimizer = eval(cfg.optim.classname)

    if cfg.validation == True:
        server_dataloader = DataLoader(
            test_data, num_workers=0, batch_size=cfg.batch_size
        )
    else:
        server_dataloader = None

    server = eval(cfg.fed.servername)(
        model, num_clients, cfg.device, dataloader=server_dataloader
    )
    clients = [
        eval(cfg.fed.clientname)(
            k,
            model,
            optimizer,
            cfg.optim.args,
            DataLoader(
                datasets[k], num_workers=0, batch_size=cfg.batch_size, shuffle=True
            ),
            cfg.device,
            **cfg.fed.args,
        )
        for k in range(num_clients)
    ]
    local_states = OrderedDict()

    for t in range(num_epochs):
        global_state = server.model.state_dict()
        for client in clients:
            client.model.load_state_dict(global_state)

        for k, client in enumerate(clients):
            client.model.load_state_dict(global_state)
            client.update()
            local_states[k] = client.model.state_dict()

        server.update(global_state, local_states)
        if cfg.validation == True:
            test_loss, accuracy = server.validation()
            log.info(
                f"[Round: {t+1: 04}] Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%"
            )


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    
    run_serial(cfg)
    
  
if __name__ == "__main__":
    main()

# To run CUDA-aware MPI:
# mpiexec -np 5 --mca opal_cuda_support 1 python ./run.py

# mpiexec -np 5 python ./run.py
