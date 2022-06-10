from omegaconf import DictConfig
from funcx import FuncXClient
import numpy as np
import torch.nn as nn
import copy
import ipdb
import time

from .algorithm import *
from .misc import *

from .funcx import client_training, client_validate_data
from .funcx import APPFLFuncTrainingEndpoints

def run_server(
    cfg: DictConfig, 
    model: nn.Module,
    loss_fn: nn.Module,
    fxc: FuncXClient,
    test_data: Dataset = Dataset()
    ):
    """ Training initialization """
    ## Logger for a server
    logger = logging.getLogger(__name__)
    logger = create_custom_logger(logger, cfg)

    ## assign number of clients
    cfg.num_clients = len(cfg.clients)

    ## funcX - APPFL training client
    trn_endps = APPFLFuncTrainingEndpoints(cfg, fxc, logger)
    
    ## Using tensorboard to visualize the test loss
    if cfg.use_tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(
            comment=cfg.fed.args.optim + "_clients_nums_" + str(cfg.num_clients)
        )
        
    ## Testing/validation variables
    test_loss = 0.0
    test_accuracy = 0.0
    best_accuracy = 0.0
    # TODO: what is comm_size?
    cfg["logginginfo"]["comm_size"] = 1

    """ Checking data at clients """
    ## Geting the total number of data samples at clients
    trn_endps.send_task_to_clients(client_validate_data)
    training_size_at_client = trn_endps.receive_sync_endpoints_updates()
    
    total_num_data = 0
    for k in range(cfg.num_clients):
        total_num_data += training_size_at_client[k]
        #TODO: What if a client doesn't have any training samples
        logger.info("Client %s has %d training samples" % (cfg.clients[k].name, training_size_at_client[k]))
        
    ## weight calculation
    weights = {}
    for k in range(cfg.num_clients):
        weights[k] = training_size_at_client[k] / total_num_data

    """ APPFL server """
    server  = eval(cfg.fed.servername)(
        weights, copy.deepcopy(model), loss_fn, cfg.num_clients, cfg.server.device, **cfg.fed.args        
    )
    # Send server model to device
    server.model.to(cfg.server.device)

    """ Server test-set data loader"""
    if cfg.validation == True and len(test_data) > 0:
        test_dataloader = DataLoader(
            test_data,
            num_workers=cfg.num_workers,
            batch_size=cfg.test_data_batch_size,
            shuffle=cfg.test_data_shuffle,
        )
    else:
        cfg.validation = False
    
    # do_continue = True

    # client_training(cfg, 0, weights, model.state_dict(), loss_fn)
    
    """ Looping over all epochs """ 
    start_time = time.time()
    for t in range(cfg.num_epochs):
        logger.info(" ====== Epoch [%d/%d] ====== " % (t+1, cfg.num_epochs))
        per_iter_start = time.time()
        
        """ Training """
        ## Get current global state
        global_state = server.model.state_dict()
        
        local_update_start = time.time()
        ## Boardcast global state and start training at funcX endpoints
        tasks   = trn_endps.send_task_to_clients(client_training,
                    weights, global_state, loss_fn)
    
        ## Aggregate local updates from clients
        local_states = []
        local_states.append(trn_endps.receive_sync_endpoints_updates())
        # TODO: timming for each client updates
        cfg["logginginfo"]["LocalUpdate_time"] = time.time() - local_update_start

        ## Perform global update
        global_update_start = time.time()
        server.update(local_states)
        cfg["logginginfo"]["GlobalUpdate_time"] = time.time() - global_update_start

        """ Validation """
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
        cfg["logginginfo"]["PerIter_time"]    = time.time() - per_iter_start
        cfg["logginginfo"]["Elapsed_time"]    = time.time() - start_time
        cfg["logginginfo"]["test_loss"]       = test_loss
        cfg["logginginfo"]["test_accuracy"]   = test_accuracy
        cfg["logginginfo"]["BestAccuracy"]    = best_accuracy

        server.logging_iteration(cfg, logger, t)

        """ Saving model"""
        if (t + 1) % cfg.checkpoints_interval == 0 or t + 1 == cfg.num_epochs:
            if cfg.save_model == True:
                save_model_iteration(t + 1, server.model, cfg)

    server.logging_summary(cfg, logger)
