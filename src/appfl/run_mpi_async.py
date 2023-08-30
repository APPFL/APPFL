import copy
import time
import logging
import torch.nn as nn
from .misc import *
from mpi4py import MPI
from typing import Any
from .algorithm import *
from torch.optim import *
from omegaconf import DictConfig
from .comm.mpi import MpiCommunicator
from torch.utils.data import DataLoader

def run_server(
    cfg: DictConfig,
    comm: MPI.Comm,
    model: nn.Module,
    loss_fn: nn.Module,
    num_clients: int,
    test_dataset: Dataset = Dataset(),
    dataset_name: str = "MNIST",
    metric: Any = None
):
    """
    run_server:
        Run PPFL server that updates the global model parameters in an asynchronous way
    args:
        cfg: the configuration for the FL experiment
        comm: MPI communicator
        model: neural network model to train
        loss_fn: loss function 
        num_clients: the number of clients used in PPFL simulation
        test_dataset: optional testing data. If given, validation will run based on this data
        dataset_name: optional dataset name
        metric: evaluation metric function
    """
    device = "cpu"
    communicator = MpiCommunicator(comm)

    logger = logging.getLogger(__name__)
    logger = create_custom_logger(logger, cfg)
    cfg.logginginfo.comm_size = comm.Get_size()
    cfg.logginginfo.DataSet_name = dataset_name

    "Using tensorboard to visualize the test loss."
    if cfg.use_tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(comment=cfg.fed.args.optim + "_clients_nums_" + str(cfg.num_clients))

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

    "Collect the number of data from each client and compute the weights for each client"
    num_data = communicator.gather(0, dest=0)
    total_num_data = 0
    for num in num_data:
        total_num_data += num
    weights = [num / total_num_data for num in num_data]

    "Asynchronous federated learning server"
    server = eval(cfg.fed.servername)(weights, copy.deepcopy(model), loss_fn, num_clients, device,**cfg.fed.args)

    start_time = time.time()
    iter = 0
    client_model_step = {i : 0 for i in range(0, num_clients)}
    client_start_time = {i : start_time for i in range(0, num_clients)}

    server.model.to("cpu")
    global_model = server.model.state_dict()

    "First broadcast the global model "
    communicator.broadcast_global_model(global_model)

    "Main server training loop"
    test_loss, test_accuracy, best_accuracy = 0.0, 0.0, 0.0
    while True:
        iter += 1
        client_idx, local_model = communicator.recv_local_model_from_client()
        logger.info(f"[Server Log] [Step #{iter:3}] Receive model from client {client_idx}")
        local_update_start = client_start_time[client_idx]
        local_update_time = time.time() - client_start_time[client_idx]
        global_update_start = time.time()
        
        server.update(local_model, client_model_step[client_idx], client_idx)
        global_update_time = time.time() - global_update_start
        if iter < cfg.num_epochs:
            client_model_step[client_idx] = server.global_step
            client_start_time[client_idx] = time.time()
            communicator.send_global_model_to_client(server.model.state_dict(), {'done': False}, client_idx) 
        # Do server validation
        validation_start = time.time()
        if cfg.validation == True:
            test_loss, test_accuracy = validation(server, test_dataloader, metric)
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
            if cfg.use_tensorboard:
                # Add them to tensorboard
                writer.add_scalar("server_test_accuracy", test_accuracy, iter)
                writer.add_scalar("server_test_loss", test_loss, iter)
        cfg.logginginfo.Validation_time = time.time() - validation_start
        cfg.logginginfo.PerIter_time = time.time() - local_update_start
        cfg.logginginfo.Elapsed_time = time.time() - start_time
        cfg.logginginfo.test_loss = test_loss
        cfg.logginginfo.test_accuracy = test_accuracy
        cfg.logginginfo.BestAccuracy = best_accuracy
        cfg.logginginfo.LocalUpdate_time = local_update_time
        cfg.logginginfo.GlobalUpdate_time = global_update_time
        logger.info(f"[Server Log] [Step #{iter:3}] Iteration Logs:")
        if iter != 1:
            logger.info(server.log_title())
        server.logging_iteration(cfg, logger, iter-1)

        "Break after max updates"
        if iter == cfg.num_epochs: 
            break
    
    "Notify the clients about the end of the learning"
    communicator.cleanup()
    for i in range(num_clients):
        communicator.send_global_model_to_client(None, {'done': True}, i)

    "Log the summary of the FL experiments"
    server.logging_summary(cfg, logger)

def run_client(
    cfg: DictConfig,
    comm: MPI.Comm,
    model: nn.Module,
    loss_fn: nn.Module,
    num_clients: int,
    train_data: Dataset,
    test_data: Dataset = Dataset(),
    metric: Any = None
):
    """
    run_client:
        Run PPFL simulation clients, each of which updates its own local parameters of model.
        [Note: For asynchronous cases, each MPI process only corresponds to one federated learning client, 
        as it does not make sense if multiple clients running serially on one MPI process in asynchronous 
        federated learning settings.]

    args:
        cfg: the configuration for the FL experiment
        comm: MPI communicator
        model: neural network model to train
        loss_fn: loss function 
        num_clients: the number of clients used in PPFL simulation
        train_data: training data
        test_data: validation data
        metric: evaluation metric function
    """
    client_idx = comm.Get_rank()-1
    communicator = MpiCommunicator(comm)

    """ log for clients"""
    output_filename = cfg.output_filename + "_client_%s" % (client_idx-1)
    outfile = client_log(cfg.output_dirname, output_filename)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    c_handler = logging.StreamHandler()
    logger.addHandler(c_handler)

    num_data = len(train_data[client_idx-1])
    communicator.gather(num_data, dest=0)

    batchsize = cfg.train_data_batch_size
    if cfg.batch_training == False:
        batchsize = len(train_data[client_idx-1])

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

    client = eval(cfg.fed.clientname)(
        client_idx,
        None,                       # TODO: Now I set weights to None, I don't know why we need weights?
        copy.deepcopy(model),
        loss_fn,
        DataLoader(
            train_data[client_idx],
            num_workers=cfg.num_workers,
            batch_size=batchsize,
            shuffle=True,
            pin_memory=True,
        ),
        cfg,
        outfile,
        test_dataloader,
        metric,
        **cfg.fed.args,
    )

    while True:
        model = communicator.recv_global_model_from_server(source=0)
        if isinstance(model, tuple):
            model, done = model[0], model[1]['done']
        else:
            done = False
        if done: 
            break
        client.model.load_state_dict(model)
        client.update()
        # Compute gradient if the algorithm is gradient-based
        if cfg.fed.args.gradient_based:
            list_named_parameters = []
            for name, _ in client.model.named_parameters():
                list_named_parameters.append(name)
            local_model = {}
            for name in model:
                if name in list_named_parameters:
                    local_model[name] = model[name] - client.primal_state[name]
                else:
                    local_model[name] = client.primal_state[name]
        else:
            local_model = copy.deepcopy(client.primal_state)
        communicator.send_local_model_to_server(local_model, dest=0)
    outfile.close()
