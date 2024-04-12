from .s3_storage import CloudStorage, LargeObjectWrapper

def load_global_state(cfg, global_state, temp_dir):
    """Download the global state if it resides on S3."""
    if CloudStorage.is_cloud_storage_object(global_state):
        CloudStorage.init(cfg, temp_dir)
        global_state = CloudStorage.download_object(global_state)  
    return global_state

def send_client_state(cfg , client_state, temp_dir, local_model_key, local_model_url):
    if cfg.use_cloud_transfer == False:
        return client_state
    client_state = LargeObjectWrapper(client_state, local_model_key)
    if not client_state.can_send_directly:
        CloudStorage.init(cfg, temp_dir)
        return CloudStorage.upload_object(client_state, object_url=local_model_url, ext='pt')
    else:
        return client_state.data