def client_validate_data(
    cfg, 
    client_idx,
    mode = 'train'
    ):
    modes = [mode] if type(mode) == str else mode 
    from appfl.funcx.client_utils import get_dataset
    data_info = {}
    for m in modes: 
        dataset = get_dataset(cfg, client_idx, mode=m)
        data_info[m] = len(dataset)
    return data_info

def client_training(
    cfg, 
    client_idx,
    weights,
    global_state,
    loss_fn
    ):
    ## Import libaries
    import os.path as osp
    from appfl.algorithm.client_optimizer import ClientOptim
    from appfl.misc import client_log, get_dataloader
    from appfl.funcx.client_utils import get_dataset, load_global_state, send_client_state, get_model
    ## Load client configs
    cfg.device         = cfg.clients[client_idx].device
    cfg.output_dirname = cfg.clients[client_idx].output_dir
    ## Prepare training/validation data
    train_dataset   = get_dataset(cfg, client_idx, mode='train')
    train_dataloader= get_dataloader(cfg, train_dataset, mode='train')
    if cfg.client_do_validation:
        val_dataset    = get_dataset(cfg, client_idx, mode='val')
        val_dataloader = get_dataloader(cfg, val_dataset, mode='val') 
    
    # Get training model
    model = get_model(cfg)
    ## Prepare output directory
    output_filename = cfg.output_filename + "_client_%s" % (client_idx)
    outfile = client_log(cfg.output_dirname, output_filename)
    ## Instantiate training agent at client 
    client= eval(cfg.fed.clientname)(
            client_idx,
            weights,
            model,
            loss_fn,
            train_dataloader,
            cfg,
            outfile,
            val_dataloader, 
            **cfg.fed.args)
    
    ## Download global state
    temp_dir = osp.join(cfg.output_dirname, "tmp")
    global_state = load_global_state(cfg, global_state, temp_dir)
    ## Initial state for a client
    client.model.to(cfg.clients[client_idx].device)
    client.model.load_state_dict(global_state)
    ## Perform a client update
    client_state = client.update()
    ## Send client state to server
    return send_client_state(cfg, client_state, client_idx, temp_dir)
    
def client_testing(
    cfg, 
    client_idx,
    weights,
    global_state,
    loss_fn
    ):
    ## Import libaries
    import os.path as osp
    from appfl.misc import client_log, get_dataloader
    from appfl.algorithm.client_optimizer import ClientOptim
    from appfl.funcx.client_utils import get_dataset, load_global_state, get_model
    ## Load client configs
    cfg.device         = cfg.clients[client_idx].device
    cfg.output_dirname = cfg.clients[client_idx].output_dir
    ## Prepare testing data
    test_dataset   = get_dataset(cfg, client_idx, mode='test')
    test_dataloader= get_dataloader(cfg, test_dataset, mode='test')
    # Get training model
    model = get_model(cfg)
    ## Prepare output directory
    output_filename = cfg.output_filename + "_client_test_%s" % (client_idx)
    outfile = client_log(cfg.output_dirname, output_filename)
    ## Instantiate training agent at client 
    client= eval(cfg.fed.clientname)(
            client_idx,
            weights,
            model,
            loss_fn,
            None,
            cfg,
            outfile,
            None,
            **cfg.fed.args)
    ## Download global state
    temp_dir = osp.join(cfg.output_dirname, "tmp")
    global_state = load_global_state(cfg, global_state, temp_dir)
    ## Initial state for a client
    client.model.to(cfg.clients[client_idx].device)
    client.model.load_state_dict(global_state)
    ## Do validation
    return client.client_validation(test_dataloader)

if __name__ == '__main__':
    from omegaconf import OmegaConf
    from appfl.config import load_funcx_device_config, load_funcx_config, FuncXConfig
    cfg = OmegaConf.structured(FuncXConfig)
    load_funcx_device_config(cfg,
        "configs/clients/torchvision_clients.yaml"
        # "configs/clients/covid19_anl_uchicago.yaml"
        )
    load_funcx_config(cfg, 
        # "configs/fed_avg/funcx_fedavg_covid.yaml",
        "configs/fed_avg/funcx_fedavg_cifar10.yaml"
        )
    
    server  = eval(cfg.fed.servername)(
            self.weights, copy.deepcopy(self.model), self.loss_fn, self.cfg.num_clients, "cpu", **self.cfg.fed.args        
        )

    print(client_training(cfg, 0))
    