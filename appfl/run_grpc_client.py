import hydra
from omegaconf import DictConfig

from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim import *
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision
from torchvision.transforms import ToTensor

import copy
import numpy as np
import logging
import time

from .protos.federated_learning_pb2 import Job
from .protos.client import FLClient
from .misc.data import Dataset
from .algorithm.iadmm import *
from .algorithm.fedavg import *

def update_model_state(comm, model, round_number):
    new_state = {}
    for name in model.state_dict():
        nparray = comm.get_tensor_record(name, round_number)
        new_state[name] = torch.from_numpy(nparray)
    model.load_state_dict(new_state)

def run_client(cfg           : DictConfig,
               comm_rank     : int,
               model         : nn.Module,
               train_dataset : Dataset,
               batch_size    : int) -> None:
    logger = logging.getLogger(__name__)
    uri = cfg.server.host + ':' + str(cfg.server.port)

    ## We assume to have as many GPUs as the number of MPI processes.
    if cfg.device == "cuda":
        device = f"cuda:{comm_rank-1}"
    else:
        device = cfg.device

    optimizer = eval(cfg.optim.classname)
    fed_client = eval(cfg.fed.clientname)(
        comm_rank,
        copy.deepcopy(model),
        optimizer,
        cfg.optim.args,
        DataLoader(train_dataset,
                   num_workers=0,
                   batch_size=batch_size,
                   shuffle=cfg.train_data_shuffle),
        device,
        **cfg.fed.args)

    # Synchronize model parameters with server.
    comm = FLClient(comm_rank, uri)
    cur_round_number, job_todo = comm.get_job(Job.INIT)
    prev_round_number = 0

    while job_todo != Job.QUIT:
        if job_todo == Job.TRAIN:
            if prev_round_number != cur_round_number:
                logger.info(f"[Client ID: {comm_rank: 03} Round #: {cur_round_number: 03}] Start training")
                update_model_state(comm, fed_client.model, cur_round_number)
                prev_round_number = cur_round_number

                fed_client.update()
                comm.send_learning_results(fed_client.model.state_dict(), cur_round_number)
                logger.info(f"[Client ID: {comm_rank: 03} Round #: {cur_round_number: 03}] Trained and sent results back to the server")
            else:
                logger.info(f"[Client ID: {comm_rank: 03} Round #: {cur_round_number: 03}] Waiting for next job")
                time.sleep(5)
        cur_round_number, job_todo = comm.get_job(job_todo)
        if job_todo == Job.QUIT:
            logger.info(f"[Client ID: {comm_rank: 03} Round #: {cur_round_number: 03}] Quitting")
            # Update with the most recent weights before exit.
            update_model_state(comm, fed_client.model, cur_round_number)

if __name__ == "__main__":
    run_client()