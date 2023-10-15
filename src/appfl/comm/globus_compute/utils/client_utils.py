import torch
from appfl.misc import get_executable_func
from .s3_storage import CloudStorage, LargeObjectWrapper

def get_dataset(cfg, client_idx, mode='train'):
    assert mode in ['train', 'val', 'test']
    if 'get_data' in cfg.clients[client_idx]:
        func_call  = get_executable_func(cfg.clients[client_idx].get_data)
    else:
        func_call  = get_executable_func(cfg.get_data)
    return func_call(cfg=cfg, client_idx=client_idx, mode=mode)

def get_model(cfg):
    get_model = get_executable_func(cfg.get_model)
    ModelClass = get_model()
    return ModelClass(**cfg.model_kwargs)

def mse_loss(pred, y):
    return torch.nn.MSELoss()(pred.float(), y.float().unsqueeze(-1))

def get_loss(cfg):
    if cfg.loss == "":
        return get_executable_func(cfg.get_loss)()()
    elif cfg.loss == "CrossEntropy":
        return torch.nn.CrossEntropyLoss()
    elif cfg.loss == "MSE":
        return mse_loss
    
def get_val_metric(cfg):
    return get_executable_func(cfg.val_metric)

def load_global_state(cfg, global_state, temp_dir):
    if CloudStorage.is_cloud_storage_object(global_state):
        CloudStorage.init(cfg, temp_dir)
        global_state = CloudStorage.download_object(global_state)  
    return global_state

def send_client_state(cfg, client_state, temp_dir, local_model_key, local_model_url):
    if cfg.use_cloud_transfer == False:
        return client_state
    client_state = LargeObjectWrapper(client_state, local_model_key)
    if not client_state.can_send_directly:
        CloudStorage.init(cfg, temp_dir)
        return CloudStorage.upload_object(client_state, object_url=local_model_url, ext='pt')
    else:
        return client_state.data