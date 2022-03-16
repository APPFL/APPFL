import sys
import time
import logging
import json
import copy
## User-defined datasets
import numpy as np
import torch
import torchvision
from torchvision.transforms import ToTensor

from appfl.config import *
from appfl.misc.data import *
from models.cnn import *
import appfl.run_grpc_server as grpc_server
import appfl.run_grpc_client as grpc_client
from mpi4py import MPI

DataSet_name = "DeepCovid"
num_clients = 1
num_channel = 3  # 1 if gray, 3 if color
num_classes = 2  # number of the image classes
num_pixel = 32  # image size = (num_pixel, num_pixel)
 
def get_model(comm: MPI.Comm):
    ## User-defined model
    model = CNN(num_channel, num_classes, num_pixel)
    return model

def main():
    # read default configuration
    cfg = OmegaConf.structured(Config)

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    ## Reproducibility
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()

    # """ CNN """
    # model = get_model(comm)
    # """ Load Data """
    # train_datasets = []
    # with open("./datasets/PreprocessedData/deepcovid_train_data.json") as f:
    #     train_data_raw = json.load(f)            
    # train_datasets.append( Dataset(
    #         torch.FloatTensor(train_data_raw["x"]), torch.tensor(train_data_raw["y"])
    #     )
    #     )
    # with open("./datasets/PreprocessedData/deepcovid_test_data.json") as f:
    #     test_data_raw = json.load(f)     
    # test_dataset = Dataset(
    #     torch.FloatTensor(test_data_raw["x"]), torch.tensor(test_data_raw["y"])
    # )
       

    """ Isabelle's DenseNet (the outputs of the model are probabilities of 1 class ) """
    import imp
    MainModel = imp.load_source('MainModel', "./models/IsabelleTorch.py")
    file = "./models/IsabelleTorch.pth"         
    model = torch.load(file)        
    model.eval()
    cfg.fed.args.loss_type = "torch.nn.BCELoss()"

    with open("./datasets/PreprocessedData/deepcovid_train_data.json") as f:
        train_data_raw = json.load(f)
        
    train_datasets = []
    train_datasets.append(Dataset( torch.FloatTensor(train_data_raw["x"]), torch.FloatTensor(train_data_raw["y"]).reshape(-1,1) ))
        
    # test data 
    with open("./datasets/PreprocessedData/deepcovid_test_data.json") as f:
        test_data_raw = json.load(f)
     
    test_dataset = Dataset( torch.FloatTensor(test_data_raw["x"]), torch.FloatTensor(test_data_raw["y"]).reshape(-1,1) )


    print(
        "----------Loaded Datasets and Model----------Elapsed Time=",
        time.time() - start_time,
    )
 

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
