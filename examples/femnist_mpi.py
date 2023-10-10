import os
import time

import json
import numpy as np
import torch

from appfl.config import *
from appfl.misc.data import *
from appfl.misc.utils import *
from models.cnn import *
import appfl.run_serial as rs
import appfl.run_mpi as rm
from mpi4py import MPI
from models.utils import get_model
from dataloader.femnist_dataloader import get_femnist
import argparse

""" read arguments """

parser = argparse.ArgumentParser()

parser.add_argument("--device", type=str, default="cpu")

## dataset
parser.add_argument("--dataset", type=str, default="FEMNIST")
parser.add_argument("--num_channel", type=int, default=1)
parser.add_argument("--num_classes", type=int, default=62)
parser.add_argument("--num_pixel", type=int, default=28)
parser.add_argument("--model", type=str, default="resnet18-legacy")

## algorithm
parser.add_argument("--federation_type", type=str, default="Federated")  ## Federated, ICEADMM, IIADMM
## clients
parser.add_argument("--num_clients", type=int, default=1)
parser.add_argument("--client_optimizer", type=str, default="Adam")
parser.add_argument("--client_lr", type=float, default=1e-3)
parser.add_argument("--num_local_epochs", type=int, default=1)

## server
parser.add_argument("--server", type=str, default="ServerFedAvg")
parser.add_argument("--num_epochs", type=int, default=2)

parser.add_argument("--server_lr", type=float, required=False)
parser.add_argument("--mparam_1", type=float, required=False)
parser.add_argument("--mparam_2", type=float, required=False)
parser.add_argument("--adapt_param", type=float, required=False)

parser.add_argument("--pretrained", type=int, default=0)

args = parser.parse_args()

if torch.cuda.is_available():
    args.device = "cuda"

def main():
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # read default configuration
    cfg = OmegaConf.structured(Config)

    ## Reproducibility
    cfg.reproduce = True
    if cfg.reproduce == True:
        set_seed(1)

    start_time = time.time()
    train_datasets, test_dataset = get_femnist(args.num_pixel, args.num_channel, args.pretrained)

    if cfg.data_sanity == True:
        data_sanity_check(
            train_datasets, test_dataset, args.num_channel, args.num_pixel
        )

    args.num_clients = len(train_datasets)
    model = get_model(args)
    loss_fn = torch.nn.CrossEntropyLoss()
    print(
        "----------Loaded Datasets and Model----------Elapsed Time=",
        time.time() - start_time,
    )

    ## settings
    cfg.device = args.device
    cfg.num_clients = args.num_clients
    cfg.num_epochs = args.num_epochs

    cfg.fed = eval(args.federation_type+"()")
    if args.federation_type == "Federated":
        cfg.fed.args.optim = args.client_optimizer
        cfg.fed.args.optim_args.lr = args.client_lr
        cfg.fed.servername = args.server
        cfg.fed.args.num_local_epochs = args.num_local_epochs

    ## outputs
    cfg.use_tensorboard = True

    if comm_size > 1:
        if comm_rank == 0:
            rm.run_server(
                cfg, comm, model, loss_fn, args.num_clients, test_dataset, args.dataset
            )
        else:
            rm.run_client(cfg, comm, model, loss_fn, args.num_clients, train_datasets)
        print("------DONE------", comm_rank)
    else:
        rs.run_serial(cfg, model, loss_fn, train_datasets, test_dataset, args.dataset)


if __name__ == "__main__":
    main()


# To run CUDA-aware MPI:
# mpiexec -np 5 --mca opal_cuda_support 1 python ./femnist.py
# To run MPI:
# mpiexec -np 5 python ./femnist.py
# To run:
# python ./femnist.py
# To run with resnet pretrained weight:
# python ./femnist.py --model resnet18 --pretrained 1
