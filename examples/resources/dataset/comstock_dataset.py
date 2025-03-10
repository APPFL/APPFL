import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Union, List, Tuple, Dict
from appfl.misc.data import (
    Dataset,
    iid_partition,
)

import os

def get_datasets(
    num_clients: int,
    client_id: int,
    data_path: str,
    split_ratios: Union[List, Tuple],
    **kwargs
):
    # assert len(split_ratios) == 3
    # # assert sum(split_ratios) == 1
    # assert all(i>=0 for i in split_ratios)
    
    cum_splits = np.cumsum(split_ratios)

    home_directory = os.environ.get("HOME")
    data_path = os.path.join(home_directory, data_path)

    data_array = np.load(data_path)
    data_array = data_array['data'][:, :, 0]
    
    train = data_array[:, :int(cum_splits[0]*data_array.shape[1])]
    val = data_array[:, int(cum_splits[0]*data_array.shape[1]):int(cum_splits[1]*data_array.shape[1])]
    test = data_array[:, int(cum_splits[1]*data_array.shape[1]):]
    
    if 0 in train.shape:
        raise ValueError("Train set is empty.")
    
    val_set = None if 0 in val.shape else val
    test_set = None if 0 in test.shape else test

    mid = len(train) // 2
    first_half = train[:mid]
    second_half = train[mid:]
    
    train = [first_half, second_half]

    return train[client_id], val_set