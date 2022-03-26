import torch
from torch.utils import data


class Dataset(data.Dataset):
    """This class provides a simple way to define client dataset for supervised learning.
    This is derived from ``torch.utils.data.Dataset`` so that can be loaded to ``torch.utils.data.DataLoader``.
    Users may also create their own dataset class derived from this for more data processing steps.

    An empty ``Dataset`` class is created if no argument is given (i.e., ``Dataset()``).

    Args:
        data_input (torch.FloatTensor): optional data inputs
        data_label (torch.Tensor): optional data ouputs (or labels)
    """

    def __init__(
        self,
        data_input: torch.FloatTensor = torch.FloatTensor(),
        data_label: torch.Tensor = torch.Tensor(),
    ):
        self.data_input = data_input
        self.data_label = data_label

    def __len__(self):
        """This returns the sample size."""
        return len(self.data_label)

    def __getitem__(self, idx):
        """This returns a sample point for given ``idx``."""
        return self.data_input[idx], self.data_label[idx]


# TODO: This is very specific to certain data format.
def data_sanity_check(train_datasets, test_dataset, num_channel, num_pixel):

    ## Check if "DataLoader" from PyTorch works.
    train_dataloader = data.DataLoader(train_datasets[0], batch_size=64, shuffle=False)

    for input, label in train_dataloader:

        assert input.shape[0] == label.shape[0]
        assert input.shape[1] == num_channel
        assert input.shape[2] == num_pixel
        assert input.shape[3] == num_pixel

    test_dataloader = data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    for input, label in test_dataloader:

        assert input.shape[0] == label.shape[0]
        assert input.shape[1] == num_channel
        assert input.shape[2] == num_pixel
        assert input.shape[3] == num_pixel
