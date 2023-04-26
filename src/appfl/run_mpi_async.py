from cmath import nan

from collections import OrderedDict
import torch.nn as nn
from torch.optim import *
from torch.utils.data import DataLoader

import numpy as np
import io
from omegaconf import DictConfig

import copy
import time
import logging

from .misc import *
from .algorithm import *

from mpi4py import MPI


def run_server(
    cfg: DictConfig,
    comm: MPI.Comm,
    model: nn.Module,
    loss_fn: nn.Module,
    num_clients: int,
    test_dataset: Dataset = Dataset(),
    dataset_name: str = "appfl",
    max_updates: int = 20
):
    """Run PPFL simulation server that aggregates and updates the global parameters of model in an asynchronous way

    Args:
        cfg (DictConfig): the configuration for this run
        comm: MPI communicator
        model (nn.Module): neural network model to train
        loss_fn (nn.Module): loss function 
        num_clients (int): the number of clients used in PPFL simulation
        test_data (Dataset): optional testing data. If given, validation will run based on this data.
        DataSet_name (str): optional dataset name
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

    staness_func = {
        'name': 'polynomial',
        'args': {'a': 0.5}
    }
    # [FedAsync] Note: cfg.fed.servername = ServerFedAsynchronous at directory '/src/appfl/algorithm/server_fed_asynchronous.py'
    server = eval(cfg.fed.servername)(
        weights, copy.deepcopy(model), loss_fn, num_clients, device, staness_func=staness_func, **cfg.fed.args
    )

    start_time = time.time()

    # [FedAsync] Record total number of local updates received from clients
    global_step = 0

    # [FedAsync] Send the (initial model, finish flag) to the clients
    server.model.to("cpu")
    send_reqs = [comm.isend((server.model.state_dict(), False), dest=i) for i in range(1, num_clients+1)]

    # [FedAsync] Wait for response (local model size) from clients
    recv_reqs = [comm.irecv(source=i, tag=i) for i in range(1, num_clients+1)]

    # [FedAsync] Step for each client
    client_model_step = {i : 0 for i in range(0, num_clients)}

    # [FedAsync] Main training loop
    while True:
        # [FedAsync] Wait for results from one client
        client_idx, state_dict_size = MPI.Request.waitany(recv_reqs)
        if client_idx != MPI.UNDEFINED:
            # [FedAsync] Increment the global step
            global_step += 1
            print(f"[Step #{global_step}] Server gets the model with size {state_dict_size} from client #{client_idx}")

            # [FedAsync] Allocate a buffer to receive the byte stream
            state_dict_bytes = np.empty(state_dict_size, dtype=np.byte)

            # [FedAsync] Receive the byte stream
            comm.Recv(state_dict_bytes, source=client_idx+1, tag=client_idx+1+comm_size)

            # [FedAsync] Load the byte to state dict
            buffer = io.BytesIO(state_dict_bytes.tobytes())
            state_dict = torch.load(buffer)

            # [FedAsync] Some bugs here
            local_model = OrderedDict()
            local_model['primal'] = state_dict

            # [FedAsync] Perform global update
            server.update([local_model], client_model_step[client_idx], client_idx)

            # [FedAsync] Remove the completed request from list
            recv_reqs.pop(client_idx)
            if global_step < max_updates:
                # [FedAsync] Send new task to the client
                comm.isend((server.model.state_dict(), False), dest=client_idx+1)
                # [FedAsync] Add new receiving request to the list
                recv_reqs.insert(client_idx, comm.irecv(source=client_idx+1, tag=client_idx+1))
                # [FedAsync] Update the model step for client
                client_model_step[client_idx] = global_step

            # Do server validation
            validation_start = time.time()
            best_accuracy = 0
            if cfg.validation == True:
                test_loss, test_accuracy = validation(server, test_dataloader)
                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                # if cfg.use_tensorboard:
                #     # Add them to tensorboard
                #     writer.add_scalar("server_test_accuracy", test_accuracy, t)
                #     writer.add_scalar("server_test_loss", test_loss, t)

            cfg["logginginfo"]["Validation_time"] = time.time() - validation_start
            cfg["logginginfo"]["Elapsed_time"] = time.time() - start_time
            cfg["logginginfo"]["test_loss"] = test_loss
            cfg["logginginfo"]["test_accuracy"] = test_accuracy
            cfg["logginginfo"]["BestAccuracy"] = best_accuracy
            server.logging_iteration(cfg, logger, global_step)
            # [FedAsync] Break after max updates
            if global_step == max_updates: 
                break

    # [FedAsync] Cancel outstanding requests
    for recv_req in recv_reqs:
        recv_req.cancel()

    # [FedAsync] Send final model and a finished indicator to all clients
    send_reqs = [comm.isend((server.model.state_dict(), True), dest=i) for i in range(1, num_clients+1)]
    MPI.Request.waitall(send_reqs)
    server.logging_summary(cfg, logger)


def run_client(
    cfg: DictConfig,
    comm: MPI.Comm,
    model: nn.Module,
    loss_fn: nn.Module,
    num_clients: int,
    train_data: Dataset,
    test_data: Dataset = Dataset()
):
    """Run PPFL simulation clients, each of which updates its own local parameters of model

    Args:
        cfg (DictConfig): the configuration for this run
        comm: MPI communicator
        model (nn.Module): neural network model to train
        num_clients (int): the number of clients used in PPFL simulation
        train_data (Dataset): training data
        test_data (Dataset): testing data
    """

    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()

    num_client_groups = np.array_split(range(num_clients), comm_size - 1)

    """ log for clients"""
    outfile = {}
    for _, cid in enumerate(num_client_groups[comm_rank - 1]):
        output_filename = cfg.output_filename + "_client_%s" % (cid)
        outfile[cid] = client_log(cfg.output_dirname, output_filename)

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

    cid = num_client_groups[comm_rank - 1][0]

    # [FedAsync] Note: cfg.fed.clientname = ClientOptim at file '/src/appfl/algorithm/client_optimizer.py'
    client = eval(cfg.fed.clientname)(
        cid,
        weight[cid],
        copy.deepcopy(model),
        loss_fn,
        DataLoader(
            train_data[cid],
            num_workers=cfg.num_workers,
            batch_size=batchsize[cid],
            shuffle=cfg.train_data_shuffle,
            pin_memory=True,
        ),
        cfg,
        outfile[cid],
        test_dataloader,
        **cfg.fed.args,
    )

    while True:
        # [FedAsync] Receive model from the server
        global_state, done = comm.recv(source=0)
        if not done:
            # [FedAsync] Train the model
            client.model.load_state_dict(global_state)
            client.update()
            state_dict = copy.deepcopy(client.primal_state)
            buffer = io.BytesIO()
            torch.save(state_dict, buffer)
            state_dict_bytes = buffer.getvalue()
            # state_dict_bytes = torch.save(state_dict, torch.ByteStorage()).tobytes()
            # [FedAsync] Send the size of state dict first
            comm.send(len(state_dict_bytes), dest=0, tag=comm_rank)
            # [FedAsync] Send the state dict
            comm.Send(np.frombuffer(state_dict_bytes, dtype=np.byte), dest=0, tag=comm_rank+comm_size)
            print(f"[Client #{comm_rank-1}] sends the model back")
        else:
            break

    client.outfile.close()


