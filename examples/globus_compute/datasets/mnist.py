def get_data(
    cfg,
    client_idx: int,
    mode='train'):
    # TODO: Support other datasets, not only MNIST at client
    import torch
    import torchvision
    from   torchvision.transforms import ToTensor
    import numpy as np
    import os.path as osp
    
    from appfl.misc.data import Dataset

    ## Prepare local dataset directory
    data_dir       = cfg.clients[client_idx].data_dir
    local_dir      = osp.join(data_dir,"RawData")
    data_raw = torchvision.datasets.MNIST(
        local_dir, download = True, 
        train = True if mode == 'train' else False, 
        transform= ToTensor()
    )
    
    split_train_data_raw = np.array_split(range(len(data_raw)),  cfg.num_clients)
    data_input = []
    data_label = []
    
    for idx in split_train_data_raw[client_idx]:
        data_input.append(data_raw[idx][0].tolist())
        data_label.append(data_raw[idx][1])
    
    return Dataset(
            torch.FloatTensor(data_input),
            torch.tensor(data_label),
        )
        