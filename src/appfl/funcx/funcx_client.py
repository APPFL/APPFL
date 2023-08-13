def client_validate_data(cfg, client_idx, mode="train", export_data_stats = False):
    modes = [mode] if type(mode) == str else mode

    ## Prepare logger
    from appfl.misc.logging import ClientLogger

    cli_logger = ClientLogger()
    cli_logger.mark_event("start_endpoint_execution")
    from appfl.funcx.client_utils import get_dataset

    data_info = {}
    data_stats = {}
    for m in modes:
        cli_logger.start_timer("load_%s_dataset" % m)
        dataset = get_dataset(cfg, client_idx, mode=m)
        data_info[m] = len(dataset)
        cli_logger.stop_timer("load_%s_dataset" % m)
        if export_data_stats == True:
            data_stats[m] = dataset.run_stats()
        else:
            data_stats = None
    cli_logger.mark_event("stop_endpoint_execution")
    return (data_info, data_stats), cli_logger.to_dict()

def client_check_cuda(cfg, client_idx,):
    import torch
    return torch.cuda.device_count()

def client_training(
    cfg,
    client_idx,
    weights,
    global_state,
    loss_fn,
    do_validation=False,
):
    from appfl.misc.logging import ClientLogger

    ## Prepare logger
    cli_logger = ClientLogger()
    cli_logger.mark_event("start_endpoint_execution")

    ## Import libaries
    import os.path as osp
    from appfl.algorithm.client_optimizer import ClientOptim
    from appfl.algorithm.funcx_client_optimizer import FuncxClientOptim
    from appfl.misc import client_log, get_dataloader
    from appfl.funcx.client_utils import (
        get_dataset,
        load_global_state,
        send_client_state,
        get_model,
    )

    ## Load client configs
    cfg.device = cfg.clients[client_idx].device
    cfg.output_dirname = cfg.clients[client_idx].output_dir

    cli_logger.start_timer("load_dataset")
    ## Prepare training/validation data
    train_dataset = get_dataset(cfg, client_idx, mode="train")
    train_dataloader = get_dataloader(cfg, train_dataset, mode="train")
    if do_validation:
        val_dataset = get_dataset(cfg, client_idx, mode="val")
        val_dataloader = get_dataloader(cfg, val_dataset, mode="val")
    else:
        val_dataloader = None
    cli_logger.stop_timer("load_dataset")

    # Get training model
    model = get_model(cfg)
    ## Prepare output directory
    output_filename = cfg.output_filename + "_client_%s" % (client_idx)
    outfile = client_log(cfg.output_dirname, output_filename)

    ## Instantiate training agent at client
    client = eval(cfg.fed.clientname)(
        client_idx,
        weights,
        model,
        loss_fn,
        train_dataloader,
        cfg,
        outfile,
        val_dataloader,
        **cfg.fed.args
    )

    ## Download global state
    cli_logger.start_timer("download_global_state")
    temp_dir = osp.join(cfg.output_dirname, "tmp")
    global_state = load_global_state(cfg, global_state, temp_dir)
    cli_logger.stop_timer("download_global_state")

    ## Initial state for a client
    cli_logger.start_timer("load_global_state_to_device")
    client.model.load_state_dict(global_state)
    cli_logger.stop_timer("load_global_state_to_device")

    ## Perform a client update
    cli_logger.start_timer("training_client_update")
    client_state, cli_logger = client.update(cli_logger=cli_logger)
    cli_logger.stop_timer("training_client_update")

    ## Send client state to server
    cli_logger.start_timer("upload_client_state")
    res = send_client_state(cfg, client_state, client_idx, temp_dir)
    cli_logger.stop_timer("upload_client_state")

    cli_logger.mark_event("stop_endpoint_execution")
    return res, cli_logger.to_dict()


def client_testing(cfg, client_idx, weights, global_state, loss_fn):
    from appfl.misc.logging import ClientLogger
    from appfl.algorithm.funcx_client_optimizer import FuncxClientOptim

    ## Prepare logger
    cli_logger = ClientLogger()
    cli_logger.mark_event("start_endpoint_execution")
    ## Import libaries
    import os.path as osp
    from appfl.misc import client_log, get_dataloader
    from appfl.algorithm.client_optimizer import ClientOptim
    from appfl.funcx.client_utils import get_dataset, load_global_state, get_model

    ## Load client configs
    cfg.device = cfg.clients[client_idx].device
    cfg.output_dirname = cfg.clients[client_idx].output_dir

    ## Prepare testing data
    cli_logger.start_timer("load_dataset")
    test_dataset = get_dataset(cfg, client_idx, mode="test")
    test_dataloader = get_dataloader(cfg, test_dataset, mode="test")
    cli_logger.stop_timer("load_dataset")
    # Get training model
    model = get_model(cfg)
    ## Prepare output directory
    output_filename = cfg.output_filename + "_client_test_%s" % (client_idx)
    outfile = client_log(cfg.output_dirname, output_filename)
    ## Instantiate training agent at client
    client = eval(cfg.fed.clientname)(
        client_idx, weights, model, loss_fn, None, cfg, outfile, None, **cfg.fed.args
    )
    ## Download global state
    temp_dir = osp.join(cfg.output_dirname, "tmp")
    cli_logger.start_timer("download_global_state")
    global_state = load_global_state(cfg, global_state, temp_dir)
    cli_logger.start_timer("download_global_state")

    ## Initial state for a client
    cli_logger.start_timer("load_global_state_to_device")
    client.model.to(cfg.clients[client_idx].device)
    client.model.load_state_dict(global_state)
    cli_logger.stop_timer("load_global_state_to_device")

    ## Do validation
    cli_logger.start_timer("do_testing")
    res = client.client_validation(test_dataloader)
    cli_logger.start_timer("do_testing")
    return res, cli_logger.to_dict()

def client_adapt_testing(cfg, client_idx, weights, global_state, loss_fn, mode, step = "adapt", unsupervised = True):
    assert step in ["adapt", "test"]

    from appfl.misc.logging import ClientLogger
    from appfl.algorithm.funcx_client_optimizer import FuncxClientOptim
    from appfl.funcx.client_utils import send_client_state
    ## Prepare logger
    cli_logger = ClientLogger()
    cli_logger.mark_event("start_endpoint_execution")
    ## Import libaries
    import os.path as osp
    from appfl.misc import client_log, get_dataloader
    from appfl.algorithm.client_optimizer import ClientOptim
    from appfl.funcx.client_utils import get_dataset, load_global_state, get_model
    from torch.utils.data import DataLoader
    ## Load client configs
    cfg.device = cfg.clients[client_idx].device
    cfg.output_dirname = cfg.clients[client_idx].output_dir

    ## Prepare testing data
    test_dataset = get_dataset(cfg, client_idx, mode=mode)
    
    # Get training model
    model = get_model(cfg)
    if step == "test":
        model.prepare_model_before_testing()
        test_dataloader = get_dataloader(cfg, test_dataset, mode=mode)
    else:
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=cfg.test_data_batch_size,
            num_workers=cfg.num_workers,
            shuffle=True,
            pin_memory=True,
        )       
    ## Prepare output directory
    output_filename = cfg.output_filename + "_client_test_%s" % (client_idx)
    outfile = client_log(cfg.output_dirname, output_filename)
    ## Instantiate training agent at client
    client = eval(cfg.fed.clientname)(
        client_idx, weights, model, loss_fn, None, cfg, outfile, None, **cfg.fed.args
    )
    ## Download global state
    temp_dir = osp.join(cfg.output_dirname, "tmp")
    cli_logger.start_timer("download_global_state")
    global_state = load_global_state(cfg, global_state, temp_dir)
    cli_logger.start_timer("download_global_state")

    ## Initial state for a client
    cli_logger.start_timer("load_global_state_to_device")
    client.model.to(cfg.clients[client_idx].device)
    client.model.load_state_dict(global_state)
    cli_logger.stop_timer("load_global_state_to_device")
    ## Do adaptation
    if step == "adapt":
        ## Do testing
        cli_logger.start_timer("do_adaptation")
        client_state, loss, res = client.client_adapt(test_dataloader, adapt=True)
        print("adaptation set: ", res["auc"])
        cli_logger.start_timer("do_adaptation")
        # client_state, loss, res = client.client_adapt(test_dataloader, adapt=False)
        # print("testing set: ", res["auc"])
        ## Send client state to server
        cli_logger.start_timer("upload_client_state")
        res_cli_stages = send_client_state(cfg, client_state, client_idx, temp_dir)
        cli_logger.stop_timer("upload_client_state")
        return (res_cli_stages, res), cli_logger.to_dict()    
    else:
        cli_logger.start_timer("do_testing")
        client_state, loss, res = client.client_adapt(test_dataloader, adapt=False)
        cli_logger.start_timer("do_testing")
    return res, cli_logger.to_dict()

def client_adapt_testing_joint(cfg, client_idx, weights, global_state, 
                            loss_fn, adapt_set, test_set, step = "adapt", unsupervised = True):
    
    assert step in ["adapt", "test"]
    from appfl.misc.logging import ClientLogger
    from appfl.algorithm.funcx_client_optimizer import FuncxClientOptim
    from appfl.funcx.client_utils import send_client_state
    ## Prepare logger
    cli_logger = ClientLogger()
    cli_logger.mark_event("start_endpoint_execution")
    ## Import libaries
    import os.path as osp
    from appfl.misc import client_log, get_dataloader
    from appfl.algorithm.client_optimizer import ClientOptim
    from appfl.funcx.client_utils import get_dataset, load_global_state, get_model
    from torch.utils.data import DataLoader
    
    ## Load client configs
    cfg.device = cfg.clients[client_idx].device
    cfg.output_dirname = cfg.clients[client_idx].output_dir
    
    ## Prepare testing & adaptation datasets
    adapt_dataset = get_dataset(cfg, client_idx, mode=adapt_set)
    test_dataset = get_dataset(cfg, client_idx, mode=test_set)
    
    ## Get training model
    model = get_model(cfg)

    ## Prepare testing & adaptation dataloaders
    adapt_dataloader = DataLoader(
            adapt_dataset,
            batch_size = model.adapt_batchsize,
            num_workers = cfg.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    test_dataloader = DataLoader(test_dataset,
            batch_size = model.test_batchsize,
            num_workers = cfg.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    ## Prepare output directory
    output_filename = cfg.output_filename + "_client_test_%s" % (client_idx)
    outfile = client_log(cfg.output_dirname, output_filename)
    
    ## Instantiate training agent at client
    client = eval(cfg.fed.clientname)(
        client_idx, weights, model, loss_fn, None, cfg, outfile, None, **cfg.fed.args
    )
    ## Download global state
    temp_dir = osp.join(cfg.output_dirname, "tmp")
    cli_logger.start_timer("download_global_state")
    global_state = load_global_state(cfg, global_state, temp_dir)
    cli_logger.start_timer("download_global_state")

    ## Initial state for a client
    cli_logger.start_timer("load_global_state_to_device")
    client.model.to(cfg.clients[client_idx].device)
    client.model.load_state_dict(global_state)
    cli_logger.stop_timer("load_global_state_to_device")
    
    ## Do adaptation
    if step == "adapt":
        ## Perform adaptation step
        cli_logger.start_timer("do_adaptation")
        client_state, loss, res = client.client_adapt_test_joint(adapt_dataloader, test_dataloader, adapt=True)
        cli_logger.stop_timer("do_adaptation")
        
        ## Send client state to server
        cli_logger.start_timer("upload_client_state")
        res_cli_stages = send_client_state(cfg, client_state, client_idx, temp_dir)
        cli_logger.stop_timer("upload_client_state")
        return (res_cli_stages, res), cli_logger.to_dict()    
    else:
        ## Perform testing step
        cli_logger.start_timer("do_testing")
        client_state, loss, res = client.client_adapt_test_joint(adapt_dataloader, test_dataloader, adapt=False)
        cli_logger.stop_timer("do_testing")
        return res, cli_logger.to_dict()

from torch.utils.data import DataLoader

def client_attack(cfg, client_idx, weights, global_state, loss_fn):
    from appfl.misc.logging import ClientLogger
    from appfl.algorithm.funcx_client_optimizer import FuncxClientOptim
    from appfl.funcx.cloud_storage import CloudStorage, LargeObjectWrapper
    ## Prepare logger
    cli_logger = ClientLogger()
    cli_logger.mark_event("start_endpoint_execution")
    ## Import libaries
    import os.path as osp
    from appfl.misc import client_log, get_dataloader
    from appfl.algorithm.client_optimizer import ClientOptim
    from appfl.funcx.client_utils import get_dataset, load_global_state, get_model

    ## Load client configs
    cfg.device = cfg.clients[client_idx].device
    cfg.output_dirname = cfg.clients[client_idx].output_dir

    ## Prepare training/validation data
    train_dataset = get_dataset(cfg, client_idx, mode="train")
    train_dataloader = get_dataloader(cfg, train_dataset, mode="train")

    ## Prepare testing data
    attack_dataset = get_dataset(cfg, client_idx, mode="test")
    attack_dataloader = DataLoader(
        attack_dataset,
        batch_size=1,
        num_workers=cfg.num_workers,
        shuffle=False,
        pin_memory=True,
    )
    
    # Get training model
    model = get_model(cfg)
    ## Prepare output directory
    output_filename = cfg.output_filename + "_client_test_%s" % (client_idx)
    outfile = client_log(cfg.output_dirname, output_filename)
    
    ## Instantiate training agent at client
    client = eval(cfg.fed.clientname)(
        client_idx, weights, model, loss_fn, train_dataloader, cfg, outfile, None, **cfg.fed.args
    )
    
    ## Download global state
    temp_dir = osp.join(cfg.output_dirname, "tmp")
    global_state = load_global_state(cfg, global_state, temp_dir)
    
    ## Initial state for a client 
    client.model.to(cfg.clients[client_idx].device)
    client.model.load_state_dict(global_state)
 
    ## Do validation
    attack_info = client.client_attack(attack_dataloader)
    CloudStorage.init(cfg, temp_dir)
    return CloudStorage.upload_object(LargeObjectWrapper(attack_info, "attack_info"), ext="pkl"), cli_logger.to_dict()