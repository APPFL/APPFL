import torch.nn as nn
import hydra
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

def run_server(cfg: DictConfig,
               rank: int,
               model: nn.Module,
               test_dataset: Dataset,
               num_clients: int,
               DataSet_name: str) -> None:
    # Do not launch a server if it is already on.
    channel = grpc.insecure_channel(cfg.server.host + ':' + str(cfg.server.port))
    if grpc_server_on(channel):
        print("Server is already running . . .")
        return

    op = operator.FLOperator(cfg, model, test_dataset, num_clients)
    op.servicer = server.FLServicer(cfg.server.id, str(cfg.server.port), op)

    print("Starting the server to listen to requests from clients . . .")
    server.serve(op.servicer, max_message_size=cfg.max_message_size)
