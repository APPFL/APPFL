from appfl.funcx.cloud_storage import LargeObjectWrapper


def client_validate_data(
    cfg, 
    client_idx):
    from appfl.misc import client_log, get_executable_func
    
    if 'get_data' in cfg.clients[client_idx]:
        get_data  = get_executable_func(cfg.clients[client_idx].get_data)
    else:
        get_data  = get_executable_func(cfg.get_data)

    # Get train data
    train_data = get_data(cfg, client_idx)
    return len(train_data)

def client_training(
    cfg, 
    client_idx,
    weights,
    global_state,
    loss_fn
    ):
    ## Import libaries
    from torch.utils.data import DataLoader
    import os.path as osp
    from appfl.misc import client_log, get_executable_func
    from appfl.algorithm import ClientOptim
    from appfl.funcx.cloud_storage import CloudStorage, LargeObjectWrapper
    import pickle as pkl
    get_model = get_executable_func(cfg.get_model)
    
    if 'get_data' in cfg.clients[client_idx]:
        get_data  = get_executable_func(cfg.clients[client_idx].get_data)
    else:
        get_data  = get_executable_func(cfg.get_data)

    ## Load client configs
    cfg.device         = cfg.clients[client_idx].device
    cfg.output_dirname = cfg.clients[client_idx].output_dir

    ## Prepare training/testing data
    train_data = get_data(cfg, client_idx)

    ## Prepare output directory
    output_filename = cfg.output_filename + "_client_%s" % (client_idx)
    outfile = client_log(cfg.output_dirname, output_filename)
    
    ## Get training model
    ModelClass = get_model()
    model      = ModelClass(*cfg.model_args, **cfg.model_kwargs)
    
    ## Configure training at client
    batch_size = cfg.train_data_batch_size
    
    ## Instantiate training client 
    client= eval(cfg.fed.clientname)(
            client_idx,
            weights,
            model,
            loss_fn,
            DataLoader(
                train_data,
                num_workers = cfg.num_workers,
                batch_size  = batch_size,
                shuffle     = cfg.train_data_shuffle,
                pin_memory  = True,
            ),
            cfg,
            outfile,
            None, #TODO: support validation at client
            **cfg.fed.args,
        )
    # Upload
    ## Download global state
    if CloudStorage.is_cloud_storage_object(global_state):
        CloudStorage.init(
            cfg, 
            osp.join(cfg.clients[client_idx].output_dir, "tmp")
            )
        
        global_state = CloudStorage.download_object(global_state)    
    
    ## Initial state for a client
    client.model.to(cfg.clients[client_idx].device)
    client.model.load_state_dict(global_state)
    
    ## Perform a client update
    client_state = client.update()
    client_state = LargeObjectWrapper(client_state, "client-%d" % client_idx)
    if not client_state.can_send_directly:
        # Save client's weight to file:
        CloudStorage.init(
            cfg, 
            osp.join(cfg.clients[client_idx].output_dir, "tmp")
            )
        return CloudStorage.upload_object(client_state)
    else:
        return client_state.data

if __name__ == '__main__':
    from omegaconf import OmegaConf
    from appfl.config import load_funcx_device_config, load_funcx_config, FuncXConfig
    cfg = OmegaConf.structured(FuncXConfig)
    load_funcx_device_config(cfg,
        "configs/devices/covid19_anl_uchicago.yaml")
    load_funcx_config(cfg, 
        "configs/fed_avg/funcx_fedavg_covid.yaml")
    
    print(client_validate_data(cfg, 0))
    