def get_data(
    cfg,
    client_idx: int,
    mode='train'):
    # TODO: Support other datasets, not only MNIST at client
    import torchvision
    from   torchvision.transforms import ToTensor
    import numpy as np
    import os
    import os.path as osp
    import torch
    from appfl.misc.data import Dataset

    ## Prepare local dataset directory
    data_dir       = cfg.clients[client_idx].data_dir
    local_dir      = osp.join(data_dir,"RawData")
    data_raw = eval("torchvision.datasets." + cfg.dataset)(
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
    
def get_model():
    import torch
    import torch.nn as nn
    import math

    class CNN(nn.Module):
        def __init__(self, num_channel, num_classes, num_pixel):
            super().__init__()
            self.conv1 = nn.Conv2d(
                num_channel, 32, kernel_size=5, padding=0, stride=1, bias=True
            )
            self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True)
            self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
            self.act = nn.ReLU(inplace=True)

            ###
            ### X_out = floor{ 1 + (X_in + 2*padding - dilation*(kernel_size-1) - 1)/stride }
            ###
            X = num_pixel
            X = math.floor(1 + (X + 2 * 0 - 1 * (5 - 1) - 1) / 1)
            X = X / 2
            X = math.floor(1 + (X + 2 * 0 - 1 * (5 - 1) - 1) / 1)
            X = X / 2
            X = int(X)

            self.fc1 = nn.Linear(64 * X * X, 512)
            self.fc2 = nn.Linear(512, num_classes)

        def forward(self, x):
            x = self.act(self.conv1(x))
            x = self.maxpool(x)
            x = self.act(self.conv2(x))
            x = self.maxpool(x)
            x = torch.flatten(x, 1)
            x = self.act(self.fc1(x))
            x = self.fc2(x)
            return x
    return CNN