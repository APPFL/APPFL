import torch.nn as nn
from datetime import datetime
import os.path as osp

from enum import Enum
def get_model_size(model: nn.Module):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

def check_endpoint(fxc, endpoints):
    for endpoint in endpoints:
        print("------ Status of Endpoint %s ------" % endpoint)
        endpoint_status = fxc.get_endpoint_status(endpoint)
        print("Status       : %s" % endpoint_status['status'])
        print("Workers      : %s" % endpoint_status['logs'][0]['info']['total_workers'])
        print("Pending tasks: %s" % endpoint_status['logs'][0]['info']['pending_tasks'])

    
