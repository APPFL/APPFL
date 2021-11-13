import hydra
from omegaconf import DictConfig

from collections import OrderedDict
import torch
from torch.optim import *
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision
from torchvision.transforms import ToTensor
import numpy as np

from algorithm.fedavg import *
from models import *
import logging
import time

from protos.federated_learning_pb2 import Job
import protos.client

def update_model_state(comm, model, round_number):
    new_state = {}
    for name in model.state_dict():
        nparray = comm.get_tensor_record(name, round_number)
        new_state[name] = torch.from_numpy(nparray)
    model.load_state_dict(new_state)

@hydra.main(config_path="config", config_name="config")
def run_client(cfg: DictConfig) -> None:
    logger = logging.getLogger(__name__)
    comm_rank = cfg.client.id
    uri = cfg.server.host + ':' + str(cfg.server.port)

    if cfg.dataset.torchvision == True:
        train_data = eval("torchvision.datasets." + cfg.dataset.classname)(
            "./datasets", **cfg.dataset.args, train=True, transform=ToTensor()
        )
        dataloader = DataLoader(train_data, num_workers=0, batch_size=cfg.batch_size, shuffle=True)
    else:
        raise NotImplementedError

    model = eval(cfg.model.classname)(**cfg.model.args)
    optimizer = eval(cfg.optim.classname)

    client = eval(cfg.fed.clientname)(
        comm_rank,
        model,
        optimizer,
        cfg.optim.args,
        dataloader,
        cfg.device,
        **cfg.fed.args
    )

    # Synchronize model parameters with server.
    comm = protos.client.FLClient(comm_rank, uri)
    cur_round_number, job_todo = comm.get_job(Job.INIT)
    prev_round_number = 0

    while job_todo != Job.QUIT:
        if job_todo == Job.TRAIN:
            if prev_round_number != cur_round_number:
                update_model_state(comm, client.model, cur_round_number)
                prev_round_number = cur_round_number

                client.update()
                comm.send_learning_results(client.model.state_dict(), cur_round_number)
                logger.info(f"[Client ID: {comm_rank: 03}] Trained and sent results back to the server")
            else:
                logger.info(f"[Client ID: {comm_rank: 03}] Waiting for next job")
                time.sleep(5)
        cur_round_number, job_todo = comm.get_job(job_todo)
        if job_todo == Job.QUIT:
            # Update with the most recent weights before exit.
            update_model_state(comm, client.model, cur_round_number)

if __name__ == "__main__":
    run_client()