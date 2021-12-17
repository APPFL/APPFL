import torch
from torch._C import FloatTensor
from torch.functional import Tensor
from torch.utils import data

class Dataset(data.Dataset):
    def __init__(self, data_input: torch.FloatTensor, data_label: torch.Tensor):
        self.data_input = data_input
        self.data_label = data_label

    def __len__(self):
            return len(self.data_label)
    
    def __getitem__(self, idx):
        return self.data_input[idx], self.data_label[idx]
        