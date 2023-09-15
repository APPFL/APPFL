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

from mpi4py import MPI
from typing import Any, Union, List
import copy

import math
import torch
import io

def run_server(
    cfg: DictConfig,
    comm: MPI.Comm,
    model: nn.Module,
    loss_fn: nn.Module,
    num_clients: int,
    test_dataset: Dataset = Dataset(),
    dataset_name: str = "appfl",
    metric: Any = None
):
    """Run PPFL simulation server that aggregates and updates the global parameters of model

    Args:
        cfg (DictConfig): the configuration for this run
        comm: MPI communicator
        model (nn.Module): neural network model to train. 
        loss_fn (nn.Module): loss function 
        num_clients (int): the number of clients used in PPFL simulation
        test_data (Dataset): optional testing data. If given, validation will run based on this data.
        DataSet_name (str): optional dataset name
        metric (function with 2 inputs): the metric for measuring model goodness on the test set
    """
    ## Start
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()
    num_client_groups = np.array_split(range(num_clients), comm_size - 1)

    # FIXME: I think it's ok for server to use cpu only.
    device = "cpu"

    """ log for a server """
    logger = logging.getLogger(__name__)
    logger = create_custom_logger(logger, cfg)

    cfg["logginginfo"]["comm_size"] = comm_size
    cfg["logginginfo"]["DataSet_name"] = dataset_name

    ## Using tensorboard to visualize the test loss
    if cfg.use_tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(
            comment=cfg.fed.args.optim + "_clients_nums_" + str(cfg.num_clients)
        )

    "Run validation if test data is given or the configuration is enabled."
    if cfg.validation == True and len(test_dataset) > 0:
        test_dataloader = DataLoader(
            test_dataset,
            num_workers=cfg.num_workers,
            batch_size=cfg.test_data_batch_size,
            shuffle=cfg.test_data_shuffle,
        )
    else:
        cfg.validation = False

    # Broadcast limitation according to the procs numbers
    intmax = pow(2,31) - (8 * comm_size) - 9 # INT_MAX,2147483647, does not fit on the MPI send buffer size
    recvlimit = math.floor(intmax/(comm_size-1)) # The total recv size should be less than intmax due to the MPI recv buffer size
    recvlimit = comm.bcast(recvlimit, root=0)

    """
    Receive the number of data from clients
    Compute "weight[client] = data[client]/total_num_data" from a server
    Scatter "weight information" to clients
    """
    num_data = comm.gather(0, root=0)
    total_num_data = 0
    for rank in range(1, comm_size):
        for val in num_data[rank].values():
            total_num_data += val

    weight = []
    weights = {}
    for rank in range(comm_size):
        if rank == 0:
            weight.append(0)
        else:
            temp = {}
            for key in num_data[rank].keys():
                temp[key] = num_data[rank][key] / total_num_data
                weights[key] = temp[key]
            weight.append(temp)

    weight = comm.scatter(weight, root=0)

    # TODO: do we want to use root as a client?
    server = eval(cfg.fed.servername)(
        weights, copy.deepcopy(model), loss_fn, num_clients, device, **cfg.fed.args
    )

    do_continue = True
    start_time = time.time()
    test_loss = 0.0
    test_accuracy = 0.0
    best_accuracy = 0.0
    maxcount = -1
    counts = []
    for t in range(cfg.num_epochs):
        per_iter_start = time.time()
        do_continue = comm.bcast(do_continue, root=0)

        # We need to load the model on cpu, before communicating.
        # Otherwise, out-of-memeory error from GPU
        server.model.to("cpu")

        global_state = server.model.state_dict()

        local_update_start = time.time()
        # Sharing the maximum communication count
        global_state = comm.bcast(global_state, root=0)
        if maxcount < 0:
            counts = comm.gather(0, root=0)
            maxcount = max(counts)
            maxcount = comm.bcast(maxcount, root=0)

        # Gather 'local_states'
        local_states= [None for i in range(num_clients)]
        local_states = slicing_gather_server(comm, comm_size, local_states, counts, maxcount)

        cfg["logginginfo"]["LocalUpdate_time"] = time.time() - local_update_start
        global_update_start = time.time()
        server.update(local_states)
        cfg["logginginfo"]["GlobalUpdate_time"] = time.time() - global_update_start

        validation_start = time.time()
        if cfg.validation == True:
            test_loss, test_accuracy = validation(server, test_dataloader, metric)

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
        """ If personalization is enabled, we only save the shared layers for server, so call save_model_state_iteration.
        Otherwise, we can save the whole server model so call save_model_iteration. """
        if (t + 1) % cfg.checkpoints_interval == 0 or t + 1 == cfg.num_epochs:
            if cfg.save_model == True:
                if cfg.personalization == True:
                    save_model_state_iteration(t + 1, server.model, cfg)
                else:
                    save_model_iteration(t + 1, server.model, cfg)

        if np.isnan(test_loss) == True:
            break

    """ Summary """
    server.logging_summary(cfg, logger)

    do_continue = False
    do_continue = comm.bcast(do_continue, root=0)


def run_client(
    cfg: DictConfig,
    comm: MPI.Comm,
    model: Union[nn.Module,List],
    loss_fn: nn.Module,
    num_clients: int,
    train_data: Dataset,
    test_data: Dataset = Dataset(),
    metric: Any = None
):
    """Run PPFL simulation clients, each of which updates its own local parameters of model

    Args:
        cfg (DictConfig): the configuration for this run
        comm: MPI communicator
        model (nn.Module): if personalization is disabled, neural network model to train. if personalization is enabled, it will be a LIST 
            containing the client models (i.e. num_clients models), which can be uninitialized or preloaded with saved weights depending on user's choice to load saved model
        num_clients (int): the number of clients used in PPFL simulation
        train_data (Dataset): training data
        test_data (Dataset): testing data
        metric (function with 2 inputs): the metric for measuring model goodness on the test set
    """

    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()

    num_client_groups = np.array_split(range(num_clients), comm_size - 1)

    """ log for clients"""
    outfile = {}
    for _, cid in enumerate(num_client_groups[comm_rank - 1]):
        output_filename = cfg.output_filename + "_client_%s" % (cid)
        outfile[cid] = client_log(cfg.output_dirname, output_filename)

    # Broadcast limitation according to the procs numbers    
    recvlimit = comm.bcast(None, root=0)

    """
    Send the number of data to a server
    Receive "weight_info" from a server    
        (fedavg)            "weight_info" is not needed as of now.
        (iceadmm+iiadmm)    "weight_info" is needed for constructing coefficients of the loss_function
    """
    num_data = {}
    for _, cid in enumerate(num_client_groups[comm_rank - 1]):
        num_data[cid] = len(train_data[cid])
    
    comm.gather(num_data, root=0)

    weight = None
    weight = comm.scatter(weight, root=0)

    batchsize = {}
    for _, cid in enumerate(num_client_groups[comm_rank - 1]):
        batchsize[cid] = cfg.train_data_batch_size
        if cfg.batch_training == False:
            batchsize[cid] = len(train_data[cid])

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
        test_dataloader = None

    if "cuda" in cfg.device:
        ## Check available GPUs if CUDA is used
        num_gpu = torch.cuda.device_count()
        clientpergpu = math.ceil(num_clients/cfg.num_gpu)

    clients = []
    for _, cid in enumerate(num_client_groups[comm_rank - 1]):
        ## We assume to have as many GPUs as the number of MPI processes.
        if "cuda" in cfg.device:
            gpuindex = int(math.floor(cid/clientpergpu))
            device = f"cuda:{gpuindex}"
        else:
            device = cfg.device
        cfg.device = device
        clients.append(
            eval(cfg.fed.clientname)(
                cid,
                weight[cid],
                # deepcopy the common model if there is no personalization, else use the the clients' own model
                copy.deepcopy(model) if not cfg.personalization else model[cid],
                loss_fn,
                DataLoader(
                    train_data[cid],
                    num_workers=cfg.num_workers,
                    batch_size=batchsize[cid],
                    shuffle=cfg.train_data_shuffle,
                    pin_memory=True,
                ),
                copy.deepcopy(cfg),
                outfile[cid],
                test_dataloader,
                metric,
                **cfg.fed.args,
            )
        )
        
    do_continue = comm.bcast(None, root=0)
    
    count = -1
    maxcount = -1

    while do_continue:        
        
        """Receive "global_state" """
        global_state = comm.bcast(None, root=0)
        
        """If personalization is enabled, then delete the personalized layer weights so they
        arent overwritten in the client model when an update happens"""
        if cfg.personalization:
            keys = [key for key,_ in model[0].named_parameters()]
            for key in keys:
                if key in cfg.p_layers:
                    _ = global_state.pop(key)
                

        """ Update "local_states" based on "global_state" """
        local_states = OrderedDict()
        for client in clients:
            cid = client.id

            ## initial point for a client model
            
            """If personalization is enabled, then global_state only has partial weights (i.e. shared layers),
            so strict=False is needed to update the weights into client model"""
            if cfg.personalization:
                client.model.load_state_dict(global_state,strict=False)
            else:
                client.model.load_state_dict(global_state)

            ## client update
            update = client.update()
            
            local_states[cid] = update
        
        """ Send "local_states" to a server """
        serializedData = io.BytesIO()
        torch.save(local_states, serializedData)
        length = serializedData.getbuffer().nbytes

        if count < 0:                 
            count = math.ceil(length/recvlimit)
            comm.gather(count, root=0)
            maxcount = comm.bcast(None, root=0)
        slicing_gather_client(comm, comm_size, recvlimit, serializedData, count, maxcount, length)

        do_continue = comm.bcast(None, root=0)

    for client in clients:
        client.outfile.close()

def slicing_gather_client(comm, comm_size, recvlimit, serializedData, count, maxcount, length):
    view = serializedData.getvalue()
    for n in range(maxcount):
        if n < count:
            end = 0
            begin = n*recvlimit
            if (n+1)*recvlimit < length:
                end = (n+1)*recvlimit
            else:
                end = length
            comm.gather(view[begin:end], root=0)
        else:
            comm.gather(None, root=0)

def slicing_gather_server(comm, comm_size, local_states_out, counts, maxcount):
    recvs = {}
    for r in range(0, comm_size):
        recvs[r] = b''

    gatherindex = 0
    for n in range(maxcount):
        recv = comm.gather(None, root=0)
        for r in range(0, comm_size):
            if gatherindex < counts[r]:
                recvs[r] = b''.join([recvs[r],recv[r]])

    for rank in range(1, comm_size):
        buffer = io.BytesIO(recvs[rank])
        local_states = torch.load(buffer)
        for cid, state in local_states.items():
            local_states_out[cid] = state
    return local_states_out