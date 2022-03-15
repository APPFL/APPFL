import os
import time
import json

import numpy as np
import torch

import torchvision
from torchvision.transforms import ToTensor

from appfl.config import *
from appfl.misc.data import *
from models.cnn import *

import appfl.run as rt
from mpi4py import MPI

DataSet_name = "DeepCovid"
num_clients = 1
num_channel = 3  # 1 if gray, 3 if color
num_classes = 2  # number of the image classes
num_pixel = 32  # image size = (num_pixel, num_pixel)
 

## Run
def main():
    # read default configuration
    cfg = OmegaConf.structured(Config)

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    ## Reproducibility
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()

    
    """ Isabelle's DenseNet (the outputs of the model are probabilities of 1 class ) """
    import imp
    MainModel = imp.load_source('MainModel', "./models/IsabelleTorch.py")
    file = "./models/IsabelleTorch.pth"         
    model = torch.load(file)        
    model.eval()
    cfg.fed.args.loss_type = "torch.nn.BCELoss()"


    """ Load Data """
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
        if comm_rank == 0:
            rt.run_server(cfg, comm, model, num_clients, test_dataset, DataSet_name)
        else:
            rt.run_client(cfg, comm, model, num_clients, train_datasets)
        print("------DONE------", comm_rank)
    else:
        rt.run_serial(cfg, model, train_datasets, test_dataset, DataSet_name)


if __name__ == "__main__":
    main()


# To run CUDA-aware MPI:
# mpiexec -np 2 --mca opal_cuda_support 1 python ./deepcovid.py
# To run MPI:
# mpiexec -np 2 python ./deepcovid.py
# To run:
# python ./deepcovid.py
