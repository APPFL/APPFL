import sys
import time
import logging

## User-defined datasets
import numpy as np
import torch
import torchvision
from torchvision.transforms import ToTensor

from appfl.config import *
from appfl.misc.data import *
from appfl.misc.utils import *
from models.cnn import *
import appfl.run_grpc_server as grpc_server
import appfl.run_grpc_client as grpc_client
from mpi4py import MPI
import argparse

DataSet_name = "MNIST"
num_clients = 2
num_channel = 1  # 1 if gray, 3 if color
num_classes = 10  # number of the image classes
num_pixel = 28  # image size = (num_pixel, num_pixel)


def get_data(comm: MPI.Comm):
    comm_rank = comm.Get_rank()

    if comm_rank == 0:
        # test data for a server
        test_data_raw = eval("torchvision.datasets." + DataSet_name)(
            f"./datasets/RawData", download=True, train=False, transform=ToTensor()
        )

    comm.Barrier()
    if comm_rank > 0:
        # test data for a server
        test_data_raw = eval("torchvision.datasets." + DataSet_name)(
            f"./datasets/RawData", download=False, train=False, transform=ToTensor()
        )

    test_data_input = []
    test_data_label = []
    for idx in range(len(test_data_raw)):
        test_data_input.append(test_data_raw[idx][0].tolist())
        test_data_label.append(test_data_raw[idx][1])

    test_dataset = Dataset(
        torch.FloatTensor(test_data_input), torch.tensor(test_data_label)
    )

    # training data for multiple clients
    train_data_raw = eval("torchvision.datasets." + DataSet_name)(
        f"./datasets/RawData", download=False, train=True, transform=ToTensor()
    )

    split_train_data_raw = np.array_split(range(len(train_data_raw)), num_clients)
    train_datasets = []
    for i in range(num_clients):

        train_data_input = []
        train_data_label = []
        for idx in split_train_data_raw[i]:
            train_data_input.append(train_data_raw[idx][0].tolist())
            train_data_label.append(train_data_raw[idx][1])

        train_datasets.append(
            Dataset(
                torch.FloatTensor(train_data_input),
                torch.tensor(train_data_label),
            )
        )

    return train_datasets, test_dataset


def get_model(comm: MPI.Comm):
    ## User-defined model
    model = CNN(num_channel, num_classes, num_pixel)
    return model


def main():

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    """ Configuration """     
    cfg = OmegaConf.structured(Config)

    parser = argparse.ArgumentParser()
    parser.add_argument('--server', type=str, required=False)    
    args = parser.parse_args()   

    cfg.fed.servername = args.server

    ## Reproducibility
    if cfg.reproduce == True:
        torch.manual_seed(1)
        torch.backends.cudnn.deterministic = True 

    start_time = time.time()

    """ User-defined model """    
    model = get_model(comm)
    cfg.fed.args.loss_type = "torch.nn.CrossEntropyLoss()"  

    ## loading models 
    cfg.load_model = True
    if cfg.load_model == True:
        cfg.load_model_dirname      = "./save_models"
        cfg.load_model_filename     = "Model"               
        model = load_model(cfg)      

    """ User-defined data """        
    train_datasets, test_dataset = get_data(comm)
    
    ## Sanity check for the user-defined data
    if cfg.data_sanity == True:
        data_sanity_check(train_datasets, test_dataset, num_channel, num_pixel)        

    print(
            "--------Data and Model: Loading_Time=",
            time.time() - start_time,
        ) 
    
    """ saving models """
    cfg.save_model = False
    if cfg.save_model == True:
        cfg.save_model_dirname      = "./save_models"
        cfg.save_model_filename     = "Model"      
        cfg.save_model_checkpoints  = [2]

    if comm_size > 1:
        # Try to launch both a server and clients.
        if comm_rank == 0:
            grpc_server.run_server(cfg, model, num_clients, test_dataset)
        else:
            grpc_client.run_client(cfg, comm_rank, model, train_datasets[comm_rank - 1])
        print("------DONE------", comm_rank)
    else:
        # Just launch a server.
        grpc_server.run_server(cfg, model, num_clients, test_dataset)


if __name__ == "__main__":
    main()
