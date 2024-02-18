import os
import torch
import torchvision
from appfl.config import *
from appfl.misc.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def test_transform(dataset):
    """Return the test transformation for different datast (MNIST/CIFAR10)"""
    if dataset == "MNIST":
        return transforms.ToTensor()
    elif dataset == "CIFAR10":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return transforms.Compose([transforms.ToTensor(), normalize])
    else:
        raise NotImplementedError
    
def train_transform(dataset):
    """Return the train transformation for different datast (MNIST/CIFAR10)"""
    if dataset == "MNIST":
        return transforms.ToTensor()
    elif dataset == "CIFAR10":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),                        
            normalize,
        ])
    else:
        raise NotImplementedError

def iid_partition(train_data_raw, num_clients, visualization, output=None):
    label_counts = {}
    client_dataset_info = {}

    # training data for multiple clients
    split_train_data_raw = np.array_split(range(len(train_data_raw)), num_clients)
    train_datasets = []
    for i in range(num_clients):
        client_dataset_info[i] = {}
        train_data_input = []
        train_data_label = []
        for idx in split_train_data_raw[i]:
            train_data_input.append(train_data_raw[idx][0].tolist())
            train_data_label.append(train_data_raw[idx][1])
            label = train_data_raw[idx][1]
            if not label in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
            if not label in client_dataset_info[i]:
                client_dataset_info[i][label] = 0
            client_dataset_info[i][label] += 1
        train_datasets.append(
            Dataset(
                torch.FloatTensor(train_data_input),
                torch.tensor(train_data_label),
            )
        )
    return train_datasets

def get_mnist(
        num_clients: int,
        client_id: int,
        train_batch_size: int = 64,
        test_batch_size: int = 64,
        partition: string = "iid",
        visualization: bool = True,
        output_dirname: string = "./outputs",
        **kwargs
):

    # Get the download directory for dataset
    dir = os.getcwd() + "/datasets/RawData"

    # Root download the data if not already available.
    test_data_raw = torchvision.datasets.MNIST(dir, download=True, train=False, transform=test_transform("MNIST"))

    # Obtain the testdataset
    test_data_input = []
    test_data_label = []
    for idx in range(len(test_data_raw)):
        test_data_input.append(test_data_raw[idx][0].tolist())
        test_data_label.append(test_data_raw[idx][1])
    test_dataset = Dataset(torch.FloatTensor(test_data_input), torch.tensor(test_data_label))

    # Training data for multiple clients
    train_data_raw = torchvision.datasets.MNIST(dir, download=False, train=True, transform=train_transform("MNIST"))

    # Obtain the visualization output filename
    if visualization:
        dir = output_dirname
        if os.path.isdir(dir) == False:
            os.makedirs(dir, exist_ok=True)
        output_filename = f"MNIST_{num_clients}clients_{partition}_distribution"
        file_ext = ".pdf"
        filename = dir + "/%s%s" % (output_filename, file_ext)
        uniq = 1
        while os.path.exists(filename):
            filename = dir + "/%s_%d%s" % (output_filename, uniq, file_ext)
            uniq += 1
    else: filename = None

    # Partition the dataset
    train_datasets = iid_partition(train_data_raw, num_clients, visualization=visualization, output=filename)
    train_dataloader = DataLoader(
        train_datasets[client_id],
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    ) 
    test_dataloader = DataLoader(
        test_data_raw,
        num_workers=0,
        batch_size=test_batch_size,
        shuffle=False,
        pin_memory=True,
    )
    
    return train_dataloader, test_dataloader