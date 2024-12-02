def get_data(
    cfg,
    client_idx: int,
    mode='train'):
    import torchvision
    from   torchvision.transforms import ToTensor
    import numpy as np
    import os.path as osp
    import torch
    from appfl.misc.data import Dataset

    ## Prepare local dataset directory
    local_dir = osp.join("data","RawData")
    data_raw = eval("torchvision.datasets." + "MNIST")(
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
    