import torch
from appfl.misc.data import Dataset

def get_dummy():
    """
    Return the MNIST dataset for a given client.
    :param num_clients: total number of clients
    :param client_id: the client id
    """
    data_train = Dataset(torch.FloatTensor([[1, 2, 3]]), torch.tensor([0]))
    data_test = Dataset(torch.FloatTensor([[1, 2, 3]]), torch.tensor([0]))
    return data_train, data_test
