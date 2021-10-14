import os
from collections import OrderedDict
from torch.optim import *
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler
import torchvision
from torchvision.transforms import ToTensor
import numpy as np
from mpi4py import MPI

import hydra
from omegaconf import DictConfig

from algorithm.fedavg import *
from models import *


def run_server(cfg: DictConfig, comm):

    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()
    num_clients = cfg.num_clients

    # FIXME: I think it's ok for server to use cpu only.
    device = "cpu"

    model = eval(cfg.model.classname)(**cfg.model.args)

    if cfg.validation == True:
        if cfg.dataset.torchvision == True:
            test_data = eval("torchvision.datasets." + cfg.dataset.classname)(
                f"./datasets/{comm_rank}",
                **cfg.dataset.args,
                train=False,
                transform=ToTensor(),
            )
            dataloader = DataLoader(test_data, batch_size=cfg.batch_size)
        else:
            raise NotImplementedError
    else:
        dataloader = None

    # TODO: do we want to use root as a client?
    server = eval(cfg.fed.servername)(
        model, num_clients, device, dataloader=dataloader
    )

    do_continue = True
    local_states = OrderedDict()
    for t in range(cfg.num_epochs):
        do_continue = comm.bcast(do_continue, root=0)

        # We need to load the model on cpu, before communicating.
        # Otherwise, out-of-memeory error from GPU
        server.model.to("cpu")
        global_state = server.model.state_dict()
        global_state = comm.bcast(global_state, root=0)

        gathered_states = comm.gather(None, root=0)
        for i, states in enumerate(gathered_states):
            if states is not None:
                for sid, state in states.items():
                    local_states[sid] = state

        server.update(local_states)
        if cfg.validation == True:
            test_loss, accuracy = server.validation()
            log.info(
                f"[Round: {t+1: 04}] Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%"
            )

    do_continue = False
    do_continue = comm.bcast(do_continue, root=0)


def run_client(cfg: DictConfig, comm):

    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()
    num_clients = cfg.num_clients
    num_client_groups = np.array_split(range(num_clients), comm_size - 1)

    # We assume to have as many GPUs as the number of MPI processes.
    if cfg.device == "cuda":
        device = f"cuda:{comm_rank-1}"
    else:
        device = cfg.device

    model = eval(cfg.model.classname)(**cfg.model.args)
    optimizer = eval(cfg.optim.classname)

    if cfg.dataset.torchvision == True:
        train_data = eval("torchvision.datasets." + cfg.dataset.classname)(
            f"./datasets/{comm_rank}",
            **cfg.dataset.args,
            train=True,
            transform=ToTensor(),
        )
        dataloaders = [
            DataLoader(
                train_data,
                batch_size=cfg.batch_size,
                sampler=DistributedSampler(
                    train_data, num_replicas=num_clients, rank=cid
                ),
            )
            for cid in num_client_groups[comm_rank - 1]
        ]
    else:
        raise NotImplementedError

    clients = [
        eval(cfg.fed.clientname)(
            cid,
            model,
            optimizer,
            cfg.optim.args,
            dataloaders[i],
            device,
            **cfg.fed.args,
        )
        for i, cid in enumerate(num_client_groups[comm_rank - 1])
    ]

    do_continue = comm.bcast(None, root=0)
    local_states = OrderedDict()
    while do_continue:
        global_state = comm.bcast(None, root=0)

        # assign the globl state to the clients first (to avoid potential shallow copy)
        for client in clients:
            client.model.load_state_dict(global_state)

        for client in clients:
            client.update()

            # We need to load the model on cpu, before communicating.
            # Otherwise, out-of-memeory error from GPU
            client.model.to("cpu")
            local_states[client.id] = client.model.state_dict()

        comm.gather(local_states, root=0)
        do_continue = comm.bcast(None, root=0)


def run_serial(cfg: DictConfig):

    num_clients = cfg.num_clients
    num_epochs = cfg.num_epochs

    if cfg.dataset.torchvision == True:
        train_data = eval("torchvision.datasets." + cfg.dataset.classname)(
            "./datasets", **cfg.dataset.args, train=True, transform=ToTensor()
        )
        local_data_size = int(len(train_data) / num_clients)
        how_to_split = [local_data_size for i in range(num_clients)]
        how_to_split[-1] += len(train_data) - sum(how_to_split)
        datasets = data.random_split(train_data, how_to_split)
    else:
        raise NotImplementedError

    # print(cfg.model.classname)
    model = eval(cfg.model.classname)(**cfg.model.args)
    optimizer = eval(cfg.optim.classname)

    if cfg.validation == True:
        if cfg.dataset.torchvision == True:
            test_data = eval("torchvision.datasets." + cfg.dataset.classname)(
                "./datasets", **cfg.dataset.args, train=False, transform=ToTensor()
            )
        else:
            raise NotImplementedError

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

        server.update(local_states)
        if cfg.validation == True:
            test_loss, accuracy = server.validation()
            log.info(
                f"[Round: {t+1: 04}] Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%"
            )


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):

    comm = MPI.COMM_WORLD

    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    if comm_size > 1:
        if comm_rank == 0:
            run_server(cfg, comm)
        else:
            run_client(cfg, comm)
    else:
        run_serial(cfg)


if __name__ == "__main__":
    main()

# To run CUDA-aware MPI:
# mpiexec -np 3 --mca opal_cuda_support 1 python appfl/run.py

# mpiexec -np 3 python appfl/run.py
