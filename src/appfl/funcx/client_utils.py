from appfl.misc import get_executable_func
from appfl.funcx.cloud_storage import CloudStorage, LargeObjectWrapper


def get_dataset(cfg, client_idx, mode="train"):
    assert mode in ["train", "val", "test"], "Mode %s does not support" % mode
    if "get_data" in cfg.clients[client_idx]:
        func_call = get_executable_func(cfg.clients[client_idx].get_data)
    else:
        func_call = get_executable_func(cfg.get_data)
    return func_call(cfg, client_idx, mode)


def get_model(cfg):
    # Get training model
    get_model = get_executable_func(cfg.get_model)
    ModelClass = get_model()
    model = ModelClass(*cfg.model_args, **cfg.model_kwargs)
    return model


def load_global_state(cfg, global_state, temp_dir):
    if CloudStorage.is_cloud_storage_object(global_state):
        CloudStorage.init(cfg, temp_dir)
        global_state = CloudStorage.download_object(global_state)
    return global_state


def send_client_state(cfg, client_state, client_idx, temp_dir):
    if cfg.use_cloud_transfer == False:
        return client_state

    client_state = LargeObjectWrapper(client_state, "client-%d" % client_idx)
    if not client_state.can_send_directly:
        # Save client's weight to file:
        CloudStorage.init(cfg, temp_dir)
        return CloudStorage.upload_object(client_state, ext="pt")
    else:
        return client_state.data
