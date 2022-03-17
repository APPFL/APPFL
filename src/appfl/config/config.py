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

    # Loading Models
    load_model: bool = False
    load_model_dirname: str = ""
    load_model_filename: str = ""

    # Saving Models
    save_model: bool = False
    save_model_dirname: str = ""
    save_model_filename: str = ""
    checkpoints_interval: int = 2
    
    # FL Outputs
    output_dirname: str = "./outputs"
    output_filename: str = "result"    


    # Compute device
    device: str = "cpu"

    # Indication of whether to validate or not
    validation: bool = True

    #
    # gRPC configutations
    #

    # 100 MB for gRPC maximum message size
    max_message_size: int = 104857600

    operator: DictConfig = OmegaConf.create({"id": 1})
    server: DictConfig = OmegaConf.create(
        {
            "id": 1,
            "host": "localhost",
            "port": 50051,
            "use_tls": False,
            "api_key": None
        }
    )
    client: DictConfig = OmegaConf.create({"id": 1})
