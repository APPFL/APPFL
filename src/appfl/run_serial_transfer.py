from cmath import nan

from collections import OrderedDict
import torch.nn as nn
from torch.optim import *
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

from omegaconf import DictConfig

import copy
import time
import logging

from .misc import *
from .algorithm import *

import copy

def run_serial(
    cfg: DictConfig,
    model: nn.Module,
    loss_fn: nn.Module,
    train_data: Dataset,
    test_data: Dataset = Dataset(),
    dataset_name: str = "appfl",
    target_train_dataset: Dataset = Dataset(),
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
            comment=cfg.fed.args.optim + "_clients_nums_" + str(cfg.num_clients-1)
        )

    """ log for clients"""
    outfile = {}
    for k in range(cfg.num_clients-1):
        output_filename = cfg.output_filename + "_client_%s" % (k)
        outfile[k] = client_log(cfg.output_dirname, output_filename)

    """ weight calculation """
    total_num_data = 0
    for k in range(cfg.num_clients-1):
        if k == cfg.fed.args.target:
            continue
        total_num_data += len(train_data[k])

    weights = {}
    for k in range(cfg.num_clients-1):
        if k == cfg.fed.args.target:
            weights[k] = 0
            continue
        weights[k] = len(train_data[k]) / total_num_data

    "Run validation if test data is given or the configuration is enabled."
    test_dataloader = None
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
        weights, copy.deepcopy(model), loss_fn, cfg.num_clients-1, cfg.device, **cfg.fed.args
    )

    server.model.to(cfg.device)

    batchsize = {}
    for k in range(cfg.num_clients-1):
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
        for k in range(cfg.num_clients-1)
    ]
    
    start_time = time.time()
    test_loss = 0.0
    test_accuracy = 0.0
    best_accuracy = 0.0
    acc = []
    loss = []

    for t in range(cfg.num_epochs):
        per_iter_start = time.time()

        local_states = []

        global_state = copy.deepcopy(server.model.state_dict())

        local_update_start = time.time()
        for k, client in enumerate(clients):
            if k == cfg.fed.args.target:
                continue
            
            ## initial point for a client model            
            client.model.load_state_dict(global_state)

            ## client update
            local_states.append(client.update())

        cfg["logginginfo"]["LocalUpdate_time"] = time.time() - local_update_start

        global_update_start = time.time()
        server.update(local_states)
        cfg["logginginfo"]["GlobalUpdate_time"] = time.time() - global_update_start

        #TODO: add another update based on the new global model and the target data ######
        target_update_start = time.time()
        
        output_filename = cfg.output_filename + "_client_target" 
        outfile_ = client_log(cfg.output_dirname, output_filename)
        global_state = copy.deepcopy(server.model.state_dict())
        cfg.fed.args.optim_args.lr /= 5

        batchsize_ = cfg.train_data_batch_size
        if cfg.batch_training == False:
            batchsize_ = len(target_train_dataset)

        # cfg.fed.args.target = 0
        target_client = eval(cfg.fed.clientname)(
            cfg.fed.args.target,
            1,
            copy.deepcopy(model),
            loss_fn,
            DataLoader(
                target_train_dataset,
                num_workers=cfg.num_workers,
                batch_size=batchsize_,
                shuffle=cfg.train_data_shuffle,
                pin_memory=True,
            ),
            cfg,
            outfile_,
            test_dataloader,
            **cfg.fed.args,
        )
        # server2 = eval(cfg.fed.servername)(
        #     [1], copy.deepcopy(model), loss_fn, 1, cfg.device, **cfg.fed.args
        # )
        # server2.model.to(cfg.device)
        target_client.model.load_state_dict(global_state)
        # server2.model.load_state_dict(global_state)
        target_client.update()
        server.model.load_state_dict(copy.deepcopy(target_client.model.state_dict()))
        # cfg.fed.args.target = cfg.num_clients-1
        # global_state = server.model.state_dict()

        cfg.fed.args.optim_args.lr *= 5
        cfg["logginginfo"]["TargetUpdate_time"] = time.time() - target_update_start

        # validation
        validation_start = time.time()
        if cfg.validation == True:
            if cfg.fed.clientname == 'FedMTLClient':
                test_loss, test_accuracy = validation_MTL(server, test_dataloader)
            else:
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
        acc.append(test_accuracy)
        loss.append(test_loss)
        cfg["logginginfo"]["BestAccuracy"] = best_accuracy

        server.logging_iteration(cfg, logger, t)

        """ Saving model """
        if (t + 1) % cfg.checkpoints_interval == 0 or t + 1 == cfg.num_epochs:
            if cfg.save_model == True:
                save_model_iteration(t + 1, server.model, cfg)
        plt.plot(np.arange(len(acc)),acc)
        plt.savefig('%s_acc_%f_%d_epoch.png' %(cfg.fed.clientname, cfg.fed.args.optim_args.lr,cfg.num_epochs))
        plt.close()
        plt.plot(np.arange(len(loss)),loss)
        plt.savefig('%s_loss_%f_%d_epoch.png'%(cfg.fed.clientname, cfg.fed.args.optim_args.lr,cfg.num_epochs))
        plt.close()

    server.logging_summary(cfg, logger)
    

    for k, client in enumerate(clients):
        client.outfile.close()