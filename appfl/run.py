import os
from collections import OrderedDict
import torch
import torch.distributed as dist
from torch.optim import *
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler
import torchvision
from torchvision.transforms import ToTensor

import hydra
from omegaconf import DictConfig, OmegaConf

from algorithm.fedavg import *
from models import *


def run_server():
    pass


def run_client():
    # sampler = DistributedSampler(train_data)
    pass


def run_serial(cfg: DictConfig):

    num_clients = cfg.num_clients
    num_epochs = cfg.num_epochs

    if cfg.dataset.distributed == True:
        # TODO
        raise NameError("distributed dataset.")
    else:
        train_data = eval("torchvision.datasets." + cfg.dataset.classname)(
            "./datasets", **cfg.dataset.args,
            train=True,
            transform=ToTensor()
        )
        local_data_size = int(len(train_data) / num_clients)
        how_to_split = [local_data_size for i in range(num_clients)]
        how_to_split[-1] += len(train_data) - sum(how_to_split)
        datasets = data.random_split(train_data, how_to_split)

    # print(cfg.model.classname)
    model = eval(cfg.model.classname)(**cfg.model.args)
    optimizer = eval(cfg.optim.classname)

    if cfg.validation == True:
        test_data = eval("torchvision.datasets." + cfg.dataset.classname)(
            "./datasets", **cfg.dataset.args,
            train=False,
            transform=ToTensor()
        )
        server_dataloader = DataLoader(test_data, num_workers=0, batch_size=cfg.batch_size)
    else:
        server_dataloader = None

    server = eval(cfg.fed.servername)(model, num_clients, cfg.device, dataloader=server_dataloader)
    clients = [
        eval(cfg.fed.clientname)(
            k,
            model,
            optimizer,
            cfg.optim.args,
            DataLoader(datasets[k], num_workers=0, batch_size=cfg.batch_size, shuffle=True),
            cfg.device,
            **cfg.fed.args
        )
        for k in range(num_clients)
    ]
    local_states = OrderedDict()

    for t in range(num_epochs):
        for k, client in enumerate(clients):
            client.model = server.get_model()
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
    if dist.is_available():
        dist.init_process_group("mpi")

        rank = dist.get_rank()
        size = dist.get_world_size()

        if rank == 0:
            run_server()
        else:
            run_client()
    else:
        print("torch.distributed is not available")
        # run_serial(model, dataloader)
        run_serial(cfg)


if __name__ == "__main__":
    main()

