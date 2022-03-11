from dataclasses import dataclass
from typing import Any
from omegaconf import DictConfig, OmegaConf

from .fed.federated import *
from .fed.iceadmm import *  ## TODO: combine iceadmm and iiadmm under the name of ADMM.
from .fed.iiadmm import *

@dataclass
class Config:
    fed: Any = Federated()

    # Compute device
    device: str = "cuda"    

    # Number of training epochs
    num_epochs: int = 2
    
    # Train data batch info
    batch_training: bool = True  ## TODO: revisit
    train_data_batch_size: int = 64
    train_data_shuffle: bool = False

    # Indication of whether to validate or not using testing data
    validation: bool = True    
    test_data_batch_size: int = 64
    test_data_shuffle: bool = False

    # Checking data sanity
    data_sanity: bool = False

    # Reproducibility
    reproduce: bool = True

    # Loading models
    load_model: bool = False
    load_model_dirname: str = ""
    load_model_filename: str = ""

    # Saving models
    save_model: bool = True
    save_model_dirname: str = "./save_models"
    save_model_filename: str = "Model"    
    
    # Logging and recording outputs
    output_dirname: str = "./outputs"
    output_filename: str = "result"    
    logginginfo: DictConfig = OmegaConf.create({})
  
    
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
            "port": 50051,
            "use_tls": False,
            "api_key": None
        }
    )
    client: DictConfig = OmegaConf.create({"id": 1})
