from omegaconf import DictConfig
from funcx import FuncXClient
import numpy as np
import torch.nn as nn
from .misc import *
from .funcx import APPFLFuncXServer
import ipdb

from appfl.funcx import client_training
import time

def run_server(
    cfg: DictConfig, 
    model: nn.Module,
    loss_fn: nn.Module,
    fxc: FuncXClient
    ):
    srvr   = APPFLFuncXServer(cfg, fxc)
    ## Start training at clients
    tasks  = srvr.send_task_to_clients(client_training,
                model.state_dict(), loss_fn)
    ## Revieve local updates from clients
    results = srvr.receive_sync_client_updates()
    print("Final results: ", results)

    
    
    
    

