import abc
from http import client
from appfl.funcx.cloud_storage import LargeObjectWrapper
from omegaconf import DictConfig
from funcx import FuncXClient
import numpy as np
import torch.nn as nn
import copy
import time
from ..algorithm import *
from ..misc import *

from .funcx_client import client_training, client_testing, client_validate_data
from .funcx_sync_server import APPFLFuncXSyncServer

class APPFLFuncXTestServer(APPFLFuncXSyncServer):
    def _run_sync_task(self, func, *args, **kwargs):
        _args=[args[i].data if type(args[i]) == LargeObjectWrapper else args[i] for i in range(len(args))]
        _kwargs={k: kwargs[k].data if type(kwargs[k]) == LargeObjectWrapper else kwargs[k] for k in kwargs}
        client_results    = OrderedDict()
        client_logs       = OrderedDict()
        self.cfg.use_cloud_transfer = False
        client_results[0],client_logs[0] = func(self.cfg, 0, *_args, **_kwargs)
        return client_results, client_logs