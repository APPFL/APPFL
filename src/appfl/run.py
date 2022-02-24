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

from .misc.data import Dataset
from .misc.utils import *

from .algorithm.fedavg import *
from .algorithm.iceadmm import *
from .algorithm.iiadmm import *

from mpi4py import MPI

def run_serial(
    cfg: DictConfig,
    model: nn.Module,
    train_data: Dataset,
    test_data: Dataset = Dataset(),
    DataSet_name: str = "appfl",
):
    """Run serial simulation of PPFL.

    Args:
        cfg (DictConfig): the configuration for this run
        model (nn.Module): neural network model to train
        train_data (Dataset): training data
        test_data (Dataset): optional testing data. If given, validation will run based on this data.
        DataSet_name (str): optional dataset name
    """

    logger = logging.getLogger(__name__)
    logger = create_custom_logger(logger, cfg)    
    title = log_title()    
    logger.info(title)

    num_clients = len(train_data)
    num_epochs = cfg.num_epochs

    """ weight calculation """    
    total_num_data = 0    
    for k in range(num_clients):
        total_num_data += len( train_data[k] )

    weights={}
    for k in range(num_clients):        
        weights[k] = len(train_data[k]) / total_num_data
        
    "Run validation if test data is given or the configuration is enabled."
    if cfg.validation == True and len(test_data) > 0:
        server_dataloader = DataLoader(
            test_data,
            num_workers=0,
            batch_size=cfg.test_data_batch_size,
            shuffle=cfg.test_data_shuffle,
        )
    else:
        cfg.validation = False

    server = eval(cfg.fed.servername)(
        weights, copy.deepcopy(model), num_clients, cfg.device, **cfg.fed.args
    )

    server.model.to(cfg.device)

    batchsize = {}
    for k in range(num_clients):
        batchsize[k] = cfg.train_data_batch_size
        if cfg.batch_training == False:
            batchsize[k] = len(train_data[k])

    clients = [
        eval(cfg.fed.clientname)(
            k,
            weights[k],
            copy.deepcopy(model),
            DataLoader(
                train_data[k],
                num_workers=0,
                batch_size=batchsize[k],
                shuffle=cfg.train_data_shuffle,
            ),
            cfg.device,
            **cfg.fed.args,
        )
        for k in range(num_clients)
    ]

    
    """ Loading Model """
    if cfg.load_model == True:      
        server.model = load_model(cfg)        

    local_states = []
    local_state = OrderedDict()
    local_state[0] = OrderedDict()

    start_time = time.time()
    test_loss = 0.0
    accuracy = 0.0
    BestAccuracy = 0.0
    for t in range(num_epochs):
        PerIter_start = time.time()

        global_state = server.model.state_dict()
        LocalUpdate_start = time.time()
        for k, client in enumerate(clients):            
            client.model.load_state_dict(global_state)                        
            local_state[0][k] = client.update()      
        
        local_states.append(local_state[0])   
 
        LocalUpdate_time = time.time() - LocalUpdate_start
        
        GlobalUpdate_start = time.time()
        prim_res, dual_res, rho_min, rho_max = server.update(local_states)
        GlobalUpdate_time = time.time() - GlobalUpdate_start

        Validation_start = time.time()
        if cfg.validation == True:
            test_loss, accuracy = validation(server, server_dataloader)
            if accuracy > BestAccuracy:
                BestAccuracy = accuracy
        Validation_time = time.time() - Validation_start
        PerIter_time = time.time() - PerIter_start
        Elapsed_time = time.time() - start_time

        log_iter = log_iteration(            
            t,
            LocalUpdate_time,
            GlobalUpdate_time,
            Validation_time,
            PerIter_time,
            Elapsed_time,
            test_loss,
            accuracy,
            prim_res, 
            dual_res,
            rho_min, 
            rho_max,
        )
        logger.info(log_iter)

    log_summary(
        logger, cfg, 1, DataSet_name, num_clients, Elapsed_time, BestAccuracy
    )

    """ Saving model """    
    if cfg.save_model == True:        
        save_model(server.model, cfg)
         

def run_server(
    cfg: DictConfig,
    comm: MPI.Comm,
    model: nn.Module,
    num_clients: int,
    test_dataset: Dataset = Dataset(),
    DataSet_name: str = "appfl",
):
    """Run PPFL simulation server that aggregates and updates the global parameters of model

    Args:
        cfg (DictConfig): the configuration for this run
        comm: MPI communicator
        model (nn.Module): neural network model to train
        num_clients (int): the number of clients used in PPFL simulation
        test_data (Dataset): optional testing data. If given, validation will run based on this data.
        DataSet_name (str): optional dataset name
    """
    
    logger = logging.getLogger(__name__)
    logger = create_custom_logger(logger, cfg)    
    title = log_title()    
    logger.info(title)
 
    ## Start
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()
    num_client_groups = np.array_split(range(num_clients), comm_size - 1)

    # FIXME: I think it's ok for server to use cpu only.
    device = "cpu"

    "Run validation if test data is given or the configuration is enabled."
    if cfg.validation == True and len(test_dataset) > 0:
        server_dataloader = DataLoader(
            test_dataset,
            num_workers=0,
            batch_size=cfg.test_data_batch_size,
            shuffle=cfg.test_data_shuffle,
        )
    else:
        cfg.validation = False
    
    """
    Receive the number of data from clients
    Compute "weight[client] = data[client]/total_num_data" from a server    
    Scatter "weight information" to clients        
    """    
    Num_Data = comm.gather(0, root=0)
    total_num_data = 0
    for rank in range(1, comm_size):
        for val in Num_Data[rank].values():
            total_num_data += val             
    
    weight=[]; weights = {}
    for rank in range(comm_size):
        if rank == 0:
            weight.append(0)
        else:
            temp = {}
            for key in Num_Data[rank].keys():
                temp[key]       = Num_Data[rank][key] / total_num_data
                weights[key]    = temp[key]
            weight.append(temp)
    
    weight = comm.scatter(weight, root = 0)

    # TODO: do we want to use root as a client?    
    server = eval(cfg.fed.servername)(
        weights, copy.deepcopy(model), num_clients, device, **cfg.fed.args
    ) 
 
    """ Loading Model """
    if cfg.load_model == True:      
        server.model = load_model(cfg)        
        
    do_continue = True
    start_time = time.time()
    test_loss = 0.0
    accuracy = 0.0
    BestAccuracy = 0.0
    for t in range(cfg.num_epochs):
        PerIter_start = time.time()
        do_continue = comm.bcast(do_continue, root=0)

        # We need to load the model on cpu, before communicating.
        # Otherwise, out-of-memeory error from GPU
        server.model.to("cpu")

        global_state = server.model.state_dict()

        LocalUpdate_start = time.time()
        global_state = comm.bcast(global_state, root=0)
        local_states = comm.gather(None, root=0)
        LocalUpdate_time = time.time() - LocalUpdate_start

        GlobalUpdate_start = time.time()        
        prim_res, dual_res, rho_min, rho_max = server.update(local_states)
        GlobalUpdate_time = time.time() - GlobalUpdate_start

        Validation_start = time.time()
        if cfg.validation == True:
            test_loss, accuracy = validation(server, server_dataloader)
            if accuracy > BestAccuracy:
                BestAccuracy = accuracy
        Validation_time = time.time() - Validation_start
        PerIter_time = time.time() - PerIter_start
        Elapsed_time = time.time() - start_time

        log_iter = log_iteration(            
            t,
            LocalUpdate_time,
            GlobalUpdate_time,
            Validation_time,
            PerIter_time,
            Elapsed_time,
            test_loss,
            accuracy,
            prim_res, 
            dual_res,
            rho_min, 
            rho_max,
        )
        logger.info(log_iter)

        if np.isnan(test_loss) == True:
            break

    do_continue = False
    do_continue = comm.bcast(do_continue, root=0)

    log_summary(
        logger, cfg, comm_size, DataSet_name, num_clients, Elapsed_time, BestAccuracy
    )
 
    """ Saving model """    
    if cfg.save_model == True:        
        save_model(server.model, cfg)
         
    



    


def run_client(
    cfg: DictConfig, comm: MPI.Comm, model: nn.Module, num_clients: int, train_data: Dataset
):
    """Run PPFL simulation clients, each of which updates its own local parameters of model

    Args:
        cfg (DictConfig): the configuration for this run
        comm: MPI communicator
        model (nn.Module): neural network model to train
        num_clients (int): the number of clients used in PPFL simulation
        train_data (Dataset): testing data
    """

    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()

    ## We assume to have as many GPUs as the number of MPI processes.
    if cfg.device == "cuda":
        device = f"cuda:{comm_rank-1}"
    else:
        device = cfg.device

    num_client_groups = np.array_split(range(num_clients), comm_size - 1)

    """
    Send the number of data to a server
    Receive "weight_info" from a server    
        (fedavg)            "weight_info" is not needed as of now.
        (iceadmm+iiadmm)    "weight_info" is needed for constructing coefficients of the loss_function         
    """
    num_data = {}
    for i, cid in enumerate(num_client_groups[comm_rank - 1]):
        num_data[cid] = len(train_data[cid])
    comm.gather(num_data, root=0)        
    weight = None
    weight = comm.scatter(weight, root = 0)
    

    batchsize = {}
    for _, cid in enumerate(num_client_groups[comm_rank - 1]):
        batchsize[cid] = cfg.train_data_batch_size
        if cfg.batch_training == False:
            batchsize[cid] = len(train_data[cid])

    clients = [
        eval(cfg.fed.clientname)(
            cid,
            weight[cid],
            copy.deepcopy(model),
            DataLoader(
                train_data[cid],
                num_workers=0,
                batch_size=batchsize[cid],
                shuffle=cfg.train_data_shuffle,
            ),
            device,
            **cfg.fed.args,
        )
        for i, cid in enumerate(num_client_groups[comm_rank - 1])
    ]

    do_continue = comm.bcast(None, root=0)

    local_states = OrderedDict()

    while do_continue:
        """ Receive "global_state" """    
        global_state = comm.bcast(None, root=0)

        """ Update "local_states" based on "global_state" """        
        for client in clients:          
            client.model.load_state_dict(global_state)
            local_states[client.id] = client.update()

        """ Send "local_states" to a server """                
        comm.gather(local_states, root=0)

        do_continue = comm.bcast(None, root=0)
