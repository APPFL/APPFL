def client_get_data(
    cfg,
    client_idx: int):
    # TODO: Loading other datasets, not only MNIST at client
    import torchvision
    from   torchvision.transforms import ToTensor
    import numpy as np
    import os
    import torch
    from appfl.misc.data import Dataset

    local_dir = os.getcwd() + "/datasets/RawData%d" %client_idx
    train_data_raw = eval("torchvision.datasets." + cfg.dataset)(
        local_dir, download = True, train = True, transform= ToTensor()
    )
    split_train_data_raw = np.array_split(range(len(train_data_raw)),  100)
    train_datasets = []
    train_data_input = []
    train_data_label = []
    
    for idx in split_train_data_raw[client_idx]:
        train_data_input.append(train_data_raw[idx][0].tolist())
        train_data_label.append(train_data_raw[idx][1])
    
    train_dataset = Dataset(
            torch.FloatTensor(train_data_input),
            torch.tensor(train_data_label),
        )
    return train_dataset

def client_training(
    cfg, 
    client_idx,
    server_weight,
    loss_fn
    ):
    ## Import libaries
    import torch
    from torch.utils.data import DataLoader
    from appfl.misc import client_log
    from appfl.algorithm import ClientOptim
    from appfl.funcx import client_get_data, get_model
    
    ## Prepare training/testing data
    train_data = client_get_data(cfg, client_idx)

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
            server_weight,
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
            None,
            **cfg.fed.args,
        )
    ## Initial weight for a client
    client.model.load_state_dict(server_weight)

    ## Perform client update
    client_state = client.update()
    return client_state