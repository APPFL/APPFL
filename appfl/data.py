import torch
from torch.utils import data

class Dataset(data.Dataset):
    def __init__(self, data_input: torch.FloatTensor, data_label: torch.Tensor):
        self.data_input = data_input
        self.data_label = data_label

    def __len__(self):
            return len(self.data_label)
    
    def __getitem__(self, idx):
        return self.data_input[idx], self.data_label[idx]

def data_sanity_check(train_datasets, test_dataset, num_channel, num_pixel):

    ## Check if "DataLoader" from PyTorch works.
    train_dataloader = data.DataLoader(
        train_datasets[0], 
        batch_size=64, 
        shuffle=False)    

    for input, label in train_dataloader:
        
        assert(input.shape[0] == label.shape[0])
        assert(input.shape[1] == num_channel)
        assert(input.shape[2] == num_pixel)
        assert(input.shape[3] == num_pixel)

    test_dataloader = data.DataLoader(
        test_dataset, 
        batch_size=64, 
        shuffle=False)    

    for input, label in test_dataloader:
        
        assert(input.shape[0] == label.shape[0])
        assert(input.shape[1] == num_channel)
        assert(input.shape[2] == num_pixel)
        assert(input.shape[3] == num_pixel)   