from omegaconf import DictConfig

import torch
import torch.nn as nn
from torch.optim import *
from torch.utils.data import DataLoader

import copy
import numpy as np
import logging
import time

from appfl.algorithm.fedavg import *
from appfl.algorithm.iceadmm import *
from appfl.algorithm.iiadmm import *

from .protos.federated_learning_pb2 import Job
from .protos.client import FLClient
from .misc.data import Dataset

def update_model_state(comm, model, round_number):
    new_state = {}
    for name in model.state_dict():
        nparray = comm.get_tensor_record(name, round_number)
        new_state[name] = torch.tensor(nparray)
    model.load_state_dict(new_state)

def run_client(cfg        : DictConfig,
               comm_rank  : int,
               model      : nn.Module,
               train_data : Dataset) -> None:
    """Launch gRPC client to connect to the server specified in the configuration.

    Args:
        cfg (DictConfig): the configuration for this run
        comm_rank (int): MPI rank
        model (nn.Module): neural network model to train
        train_data (Dataset): training data
    """

    logger = logging.getLogger(__name__)
    if cfg.server.use_tls == True:
        uri = cfg.server.host
    else:
        uri = cfg.server.host + ':' + str(cfg.server.port)

    cid = comm_rank - 1

    ## We assume to have as many GPUs as the number of MPI processes.
    if cfg.device == "cuda":
        device = f"cuda:{cid}"
    else:
        device = cfg.device

    batch_size = cfg.train_data_batch_size
    if cfg.batch_training == False:
        batchsize = len(train_data)

    logger.debug(f"[Client ID: {cid: 03}] connecting to (uri,tls)=({uri},{cfg.server.use_tls}).")
    comm = FLClient(cid, uri, cfg.server.use_tls, max_message_size=cfg.max_message_size, api_key=cfg.server.api_key)

    # Retrieve its weight from a server.
    weight = -1.0
    i = 1
    logger.info(f"[Client ID: {cid: 03}] requesting weight to the server.")
    try:
        while True:
            weight = comm.get_weight(len(train_data))
            logger.debug(f"[Client ID: {cid: 03}] trial {i}, requesting weight ({weight}).")
            if weight >= 0.0:
                break
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info(f"[Client ID: {cid: 03}] terminating the client.")
        return

    if weight < 0.0:
        logger.error(f"[Client ID: {cid: 03}] weight ({weight}) retrieval failed.")
        return

    fed_client = eval(cfg.fed.clientname)(
        cid,
        weight,
        copy.deepcopy(model),
        DataLoader(train_data,
                   num_workers=0,
                   batch_size=batch_size,
                   shuffle=cfg.train_data_shuffle),
        device,
        **cfg.fed.args)

    # Start federated learning.
    cur_round_number, job_todo = comm.get_job(Job.INIT)
    prev_round_number = 0
    learning_time = 0.0
    send_time = 0.0
    cumul_learning_time = 0.0

    while job_todo != Job.QUIT:
        if job_todo == Job.TRAIN:
            if prev_round_number != cur_round_number:
                logger.info(f"[Client ID: {cid: 03} Round #: {cur_round_number: 03}] Start training")
                update_model_state(comm, fed_client.model, cur_round_number)
                logger.info(f"[Client ID: {cid: 03} Round #: {cur_round_number: 03}] Received model update from server")
                prev_round_number = cur_round_number

                time_start = time.time()
                local_state = fed_client.update()
                time_end = time.time()
                logger.info(f"[Client ID: {cid: 03} Round #: {cur_round_number: 03}] Updated local model")
                learning_time = time_end - time_start
                cumul_learning_time += learning_time
                time_start = time.time()
                comm.send_learning_results(local_state["penalty"], local_state["primal"], local_state["dual"], cur_round_number)
                time_end = time.time()
                send_time = time_end - time_start
                logger.info(f"[Client ID: {cid: 03} Round #: {cur_round_number: 03}] Trained (Elapsed %.4f) and sent results back to the server (Elapsed %.4f)", learning_time, send_time)
            else:
                logger.info(f"[Client ID: {cid: 03} Round #: {cur_round_number: 03}] Waiting for next job")
                time.sleep(5)
        cur_round_number, job_todo = comm.get_job(job_todo)
        if job_todo == Job.QUIT:
            logger.info(f"[Client ID: {cid: 03} Round #: {cur_round_number: 03}] Quitting... Learning %.4f Sending %.4f Receiving %.4f Job %.4f Total %.4f",
                        cumul_learning_time, comm.time_send_results, comm.time_get_tensor, comm.time_get_job, comm.get_comm_time())
            # Update with the most recent weights before exit.
            update_model_state(comm, fed_client.model, cur_round_number)

if __name__ == "__main__":
    run_client()