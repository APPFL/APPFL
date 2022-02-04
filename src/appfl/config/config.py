from dataclasses import dataclass
from typing import Any
from omegaconf import DictConfig, OmegaConf

from .fed.fedavg import *
from .fed.iceadmm import *
from .fed.iiadmm import *

@dataclass
class Config:
    fed: Any = FedAvg()

    # Number of training epochs
    num_epochs: int = 2
    
    # Training Data Batch Info
    batch_training: bool = True
    train_data_batch_size: int = 64
    train_data_shuffle: bool = False

    # Testing Data Batch Info
    test_data_batch_size: int = 64
    test_data_shuffle: bool = False

    # Results directory
    result_dir: str = "./results"

    # Compute device
    device: str = "cpu"

    # Indication of whether to validate or not
    validation: bool = True

    #
    # gRPC configutations
    #

    # 10 MB for gRPC maximum message size
    max_message_size: int = 10485760

    operator: DictConfig = OmegaConf.create({"id": 1})
    server: DictConfig = OmegaConf.create(
        {
            "id": 1,
            "host": "localhost",
            "port": 50051
        }
    )
    client: DictConfig = OmegaConf.create({"id": 1})
