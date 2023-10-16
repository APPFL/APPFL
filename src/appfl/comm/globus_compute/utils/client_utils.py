import os
import torch
from .utils import get_executable_func
from .s3_storage import CloudStorage, LargeObjectWrapper

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