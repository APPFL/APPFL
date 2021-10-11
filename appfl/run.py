import os
from collections import OrderedDict
import torch
import torch.distributed as dist
from torch.optim import *
from torch.utils import data
import torchvision

import hydra
from omegaconf import DictConfig, OmegaConf

from algorithm.fedavg import *
from models import *

def run_server():
    pass

def run_client():
    pass
    
def run_serial(cfg: DictConfig):

    num_clients = 10
    num_round = 5

    if cfg.dataset.distributed == True:
        # TODO
        raise NameError("distributed dataset.")
    else:
        local_datasets = eval('torchvision.datasets.' + cfg.dataset.classname)('./datasets', **cfg.dataset.args)
        local_data_size = int(len(local_datasets) / num_clients)
        how_to_split = [local_data_size for i in range(num_clients)]
        how_to_split[-1] += len(local_datasets) - sum(how_to_split)
        datasets = data.random_split(local_datasets, how_to_split)

    # print(cfg.model.classname)
    model = eval(cfg.model.classname)(**cfg.model.args)
    optimizer = eval(cfg.optim.classname)

    server = eval(cfg.fed.servername)(model, num_clients, cfg.device)
    clients = [
        eval(cfg.fed.clientname)(model, optimizer, cfg.optim.args, num_round, datasets[k], cfg.device) 
        for k in range(num_clients)
    ]
    local_states = OrderedDict()

    for t in range(num_round):
        for k, client in enumerate(clients):
            client.model = server.get_model()
            client.update()
            local_states[k] = client.get_model()
            server.update(local_states)
    
@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    if dist.is_available():
        dist.init_process_group('mpi')

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


