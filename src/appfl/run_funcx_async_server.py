from omegaconf import DictConfig
from funcx import FuncXClient
import torch.nn as nn
from .misc import Dataset
from .funcx.funcx_server import APPFLFuncXAsyncServer

def run_server(
    cfg: DictConfig, 
    model: nn.Module,
    loss_fn: nn.Module,
    fxc: FuncXClient,
    test_data: Dataset = Dataset()
    ):
    """ 
    This function runs an Async FL server object, and instantiate client tasks at funcX endpoints 
    Args:
        - cfg (DictConfig):
        - model (nn.Module)
    Returns:
    """
    serv = APPFLFuncXAsyncServer(cfg, fxc)

    serv.set_validation_dataset(test_data) 
    
    serv.run(model, loss_fn)
    