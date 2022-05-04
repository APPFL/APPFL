from cmath import nan

from collections import OrderedDict
import torch.nn as nn
from torch.optim import *
from torch.utils.data import DataLoader

import numpy as np

from omegaconf import DictConfig

import copy
import time
import logging

from .misc import *
from .algorithm import *


def run_serial(
    cfg: DictConfig,
    model: nn.Module,
    loss_fn: nn.Module,
    train_data: Dataset,
    test_data: Dataset = Dataset(),
    dataset_name: str = "appfl",
):
    """Run serial simulation of PPFL.

    Args:
        cfg (DictConfig): the configuration for this run
        model (nn.Module): neural network model to train
        loss_fn (nn.Module): loss function 
        train_data (Dataset): training data
        test_data (Dataset): optional testing data. If given, validation will run based on this data.
        dataset_name (str): optional dataset name
    """
    
    """ log for a server """
    logger = logging.getLogger(__name__)
    logger = create_custom_logger(logger, cfg)

    cfg["logginginfo"]["comm_size"] = 1
    cfg["logginginfo"]["DataSet_name"] = dataset_name

    ## Using tensorboard to visualize the test loss
    if cfg.use_tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(
            comment=cfg.fed.args.optim + "_clients_nums_" + str(cfg.num_clients)
        )

    """ log for clients"""
    outfile = {}
    for k in range(cfg.num_clients):
        output_filename = cfg.output_filename + "_client_%s" % (k)
        outfile[k] = client_log(cfg.output_dirname, output_filename)

    """ weight calculation """
    total_num_data = 0
    for k in range(cfg.num_clients):
        total_num_data += len(train_data[k])

    weights = {}
    for k in range(cfg.num_clients):
        weights[k] = len(train_data[k]) / total_num_data

    "Run validation if test data is given or the configuration is enabled."
    if cfg.validation == True and len(test_data) > 0:
        test_dataloader = DataLoader(
            test_data,
            num_workers=cfg.num_workers,
            batch_size=cfg.test_data_batch_size,
            shuffle=cfg.test_data_shuffle,
        )
    else:
        cfg.validation = False

    server = eval(cfg.fed.servername)(
        weights, copy.deepcopy(model), loss_fn, cfg.num_clients, cfg.device, **cfg.fed.args
    )

    server.model.to(cfg.device)

    batchsize = {}
    for k in range(cfg.num_clients):
        batchsize[k] = cfg.train_data_batch_size
        if cfg.batch_training == False:
            batchsize[k] = len(train_data[k])

    clients = [
        eval(cfg.fed.clientname)(
            k,
            weights[k],
            copy.deepcopy(model),
            loss_fn,
            DataLoader(
                train_data[k],
                num_workers=cfg.num_workers,
                batch_size=batchsize[k],
                shuffle=cfg.train_data_shuffle,
                pin_memory=True,
            ),
            cfg,
            outfile[k],
            test_dataloader,
            **cfg.fed.args,
        )
        for k in range(cfg.num_clients)
    ]

    ## name of parameters
    model_name = []
    for name, _ in server.model.named_parameters():
        model_name.append(name)
    
    start_time = time.time()
    test_loss = 0.0
    test_accuracy = 0.0
    best_accuracy = 0.0
    for t in range(cfg.num_epochs):
        per_iter_start = time.time()

        local_states = [OrderedDict()]

        global_state = server.model.state_dict()

        local_update_start = time.time()
        for k, client in enumerate(clients):
            
            ## initial point for a client model
            for name in server.model.state_dict():
                if name not in model_name:
                    global_state[name] = client.model.state_dict()[name]
            client.model.load_state_dict(global_state)

            ## client update
            local_states[0][k] = client.update()

        cfg["logginginfo"]["LocalUpdate_time"] = time.time() - local_update_start

        global_update_start = time.time()
        server.update(local_states)
        cfg["logginginfo"]["GlobalUpdate_time"] = time.time() - global_update_start

        validation_start = time.time()
        if cfg.validation == True:
            test_loss, test_accuracy = validation(server, test_dataloader)

            if cfg.use_tensorboard:
                # Add them to tensorboard
                writer.add_scalar("server_test_accuracy", test_accuracy, t)
                writer.add_scalar("server_test_loss", test_loss, t)

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy

        cfg["logginginfo"]["Validation_time"] = time.time() - validation_start
        cfg["logginginfo"]["PerIter_time"] = time.time() - per_iter_start
        cfg["logginginfo"]["Elapsed_time"] = time.time() - start_time
        cfg["logginginfo"]["test_loss"] = test_loss
        cfg["logginginfo"]["test_accuracy"] = test_accuracy
        cfg["logginginfo"]["BestAccuracy"] = best_accuracy

        server.logging_iteration(cfg, logger, t)

        """ Saving model """
        if (t + 1) % cfg.checkpoints_interval == 0 or t + 1 == cfg.num_epochs:
            if cfg.save_model == True:
                save_model_iteration(t + 1, server.model, cfg)

    server.logging_summary(cfg, logger)

    for k, client in enumerate(clients):
        client.outfile.close()
