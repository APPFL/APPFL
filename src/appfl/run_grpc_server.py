import logging

import torch.nn as nn
from omegaconf import DictConfig

import grpc

from .protos import server
from .protos import operator
from .misc.data import Dataset


def grpc_server_on(channel) -> bool:
    try:
        grpc.channel_ready_future(channel).result(timeout=1)
        return True
    except grpc.FutureTimeoutError:
        return False


def run_server(
    cfg: DictConfig,
    model: nn.Module,    
    loss_fn: nn.Module, 
    num_clients: int,
    test_data: Dataset = Dataset(),
) -> None:
    """Launch gRPC server to listen to the port to serve requests from clients.
    The service URI is set in the configuration.
    The server will not start training until the specified number of clients connect to the server.

    Args:
        cfg (DictConfig): the configuration for this run
        model (nn.Module): neural network model to train        
        loss_fn (nn.Module): loss function
        num_clients (int): the number of clients used in PPFL simulation
        test_data (Dataset): optional testing data. If given, validation will run based on this data.
    """

    # Do not launch a server if it is already on.
    # channel = grpc.insecure_channel(cfg.server.host + ':' + str(cfg.server.port))
    # if grpc_server_on(channel):
    #     print("Server is already running . . .")
    #     return

    op = operator.FLOperator(cfg, model, loss_fn, test_data, num_clients)
    op.servicer = server.FLServicer(cfg.server.id, str(cfg.server.port), op)

    logger = logging.getLogger(__name__)
    logger.info("Starting the server to listen to requests from clients . . .")
    server.serve(op.servicer, max_message_size=cfg.max_message_size)
