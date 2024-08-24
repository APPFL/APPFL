import os
import torch
import pathlib
from appfl.config import ClientAgentConfig
from typing import Union, Dict, OrderedDict, Any, Optional
from .s3_storage import CloudStorage, LargeObjectWrapper
from .utils import get_executable_func

def load_global_model(
    client_agent_config: ClientAgentConfig,
    global_model: Any
):
    s3_tmp_dir = str(
        pathlib.Path.home() / ".appfl" / "globus_compute" / client_agent_config.endpoint_id / client_agent_config.experiment_id
    )
    if not pathlib.Path(s3_tmp_dir).exists():
        pathlib.Path(s3_tmp_dir).mkdir(parents=True, exist_ok=True)
    if CloudStorage.is_cloud_storage_object(global_model):
        CloudStorage.init(s3_tmp_dir=s3_tmp_dir)
        global_model = CloudStorage.download_object(global_model)
    return global_model

def send_local_model(
    client_agent_config: ClientAgentConfig,
    local_model:  Union[Dict, OrderedDict, bytes],
    local_model_key: Optional[str],
    local_model_url: Optional[str],
):
    if (
        hasattr(client_agent_config.comm_configs, "globus_compute_configs") and
        client_agent_config.comm_configs.globus_compute_configs.get("s3_bucket", None) is not None
    ):
        s3_tmp_dir = str(
            pathlib.Path.home() / ".appfl" / "globus_compute" / client_agent_config.endpoint_id / client_agent_config.experiment_id
        )
        if not pathlib.Path(s3_tmp_dir).exists():
            pathlib.Path(s3_tmp_dir).mkdir(parents=True, exist_ok=True)
        local_model_wrapper = LargeObjectWrapper(local_model, local_model_key)
        if not local_model_wrapper.can_send_directly:
            CloudStorage.init(s3_tmp_dir=s3_tmp_dir)
            local_model = CloudStorage.upload_object(
                local_model_wrapper, 
                object_url=local_model_url, 
                ext='pt' if not isinstance(local_model, bytes) else 'pkl'
            )
    return local_model


## The following functions are used to support legacy code

def get_dataset(cfg, client_idx, mode='train'):
    """
    Obtain the dataset using the client provided dataloader. 
    TODO: Think about what type of rules is needed for client-provided dataloader. I think it should be `func(model, **kwargs)`
    """
    assert mode in ['train', 'val', 'test']
    if 'get_data' in cfg.clients[client_idx]:
        func_call  = get_executable_func(cfg.clients[client_idx].get_data)
    else:
        func_call  = get_executable_func(cfg.get_data)
    return func_call(cfg=cfg, client_idx=client_idx, mode=mode)

def get_model(cfg):
    """Obtain the model instance."""
    get_model = get_executable_func(cfg.get_model)
    ModelClass = get_model()
    return ModelClass(**cfg.model_kwargs)

def mse_loss(pred, y):
    return torch.nn.MSELoss()(pred.float(), y.float().unsqueeze(-1))

def get_loss(cfg):
    """Obtain the loss function instance."""
    if cfg.loss == "":
        return get_executable_func(cfg.get_loss)()()
    elif cfg.loss == "CrossEntropy":
        return torch.nn.CrossEntropyLoss()
    elif cfg.loss == "MSE":
        return mse_loss
    
def get_val_metric(cfg):
    return get_executable_func(cfg.val_metric)

def load_global_state(cfg, global_state, temp_dir):
    """Download the global state if it resides on S3."""
    if CloudStorage.is_cloud_storage_object(global_state):
        CloudStorage.init(cfg, temp_dir)
        global_state = CloudStorage.download_object(global_state)  
    return global_state

def save_global_model(cfg, global_state, save_dir):
    """Save the global state in the local file system."""
    if CloudStorage.is_cloud_storage_object(global_state):
        CloudStorage.init(cfg, save_dir)
        global_state = CloudStorage.download_object(global_state, delete_local=False)  
    else:
        os.makedirs(save_dir, exist_ok=True)
        save_file_name = save_dir + "/final_model.pt"
        uniq = 1
        while os.path.exists(save_file_name):
            save_file_name = save_dir + f"/final_model_{uniq}.pt"
            uniq += 1
        torch.save(global_state, save_file_name)

def send_client_state(cfg, client_state, temp_dir, local_model_key, local_model_url):
    if cfg.use_cloud_transfer == False:
        return client_state
    client_state = LargeObjectWrapper(client_state, local_model_key)
    if not client_state.can_send_directly:
        CloudStorage.init(cfg, temp_dir)
        return CloudStorage.upload_object(client_state, object_url=local_model_url, ext='pt')
    else:
        return client_state.data