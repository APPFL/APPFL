import torch.nn as nn
import hydra
from omegaconf import DictConfig

from .protos import server
from .protos import operator
from .misc.data import Dataset

def run_server(cfg: DictConfig,
               rank: int,
               model: nn.Module,
               test_dataset: Dataset,
               num_clients: int,
               DataSet_name: str) -> None:
    op = operator.FLOperator(cfg, model, test_dataset, num_clients)
    op.servicer = server.FLServicer(cfg.server.id, str(cfg.server.port), op)

    print("Starting the server to listen to requests from clients . . .")
    server.serve(op.servicer)
