import os
import time
from turtle import forward

import numpy as np
import torch

import torchvision
from torchvision.transforms import ToTensor

from appfl.config import *
from appfl.misc.data import *
from appfl.misc.utils import *
from models.cnn import *

import appfl.run_serial as rs
from torchvision import transforms
import argparse

""" read arguments """

parser = argparse.ArgumentParser()

parser.add_argument("--device", type=str, default="cpu")

## dataset
parser.add_argument("--dataset", type=str, default="MNIST")
parser.add_argument("--num_channel", type=int, default=1)
parser.add_argument("--num_classes", type=int, default=10)
parser.add_argument("--num_pixel", type=int, default=28)

## clients
parser.add_argument("--num_clients", type=int, default=1)
parser.add_argument("--client_optimizer", type=str, default="SGD")
parser.add_argument("--client_lr", type=float, default=1e-3)
parser.add_argument("--num_local_epochs", type=int, default=1)

## server
parser.add_argument("--server", type=str, default="ServerFedAvg")
parser.add_argument("--num_epochs", type=int, default=3)

parser.add_argument("--server_lr", type=float, required=False)
parser.add_argument("--mparam_1", type=float, required=False)
parser.add_argument("--mparam_2", type=float, required=False)
parser.add_argument("--adapt_param", type=float, required=False)


args = parser.parse_args()

if torch.cuda.is_available():
    args.device = "cuda"

with open("../examples/models/resnet.py") as fi:
    exec(fi.read(), globals())

with open("../examples/datasets/covid19_argonne.py") as fi:
    exec(fi.read(), globals())

## Run
def main():

    """Configuration"""
    cfg = OmegaConf.structured(FuncXConfig)

    cfg.device = args.device
    cfg.reproduce = True
    if cfg.reproduce == True:
        set_seed(1)

    load_funcx_device_config(cfg, "configs/clients/covid19_anl.yaml")
    load_funcx_config(cfg, "configs/fed_avg/funcx_fedavg_covid.yaml")

    ## clients
    cfg.num_clients = args.num_clients
    cfg.fed.args.optim = args.client_optimizer
    cfg.fed.args.optim_args.lr = args.client_lr
    cfg.fed.args.num_local_epochs = args.num_local_epochs

    ## server
    cfg.fed.servername = args.server
    cfg.num_epochs = args.num_epochs

    ## outputs

    cfg.use_tensorboard = False

    cfg.save_model_state_dict = False

    cfg.output_dirname = "./outputs_%s_%s_%s" % (
        args.dataset,
        args.server,
        args.client_optimizer,
    )

    cfg.output_filename = "result"

    start_time = time.time()

    """ User-defined model """
    model = get_model()(2)
    loss_fn = torch.nn.CrossEntropyLoss()

    ## loading models
    cfg.load_model = False
    if cfg.load_model == True:
        cfg.load_model_dirname = "./save_models"
        cfg.load_model_filename = "Model"
        model = load_model(cfg)

    """ User-defined data """
    train_datasets = [get_data(cfg, 0, "train")]
    test_dataset = get_data(cfg, 0, "test")

    print(
        "-------Loading_Time=",
        time.time() - start_time,
    )

    """ saving models """
    cfg.save_model = False
    if cfg.save_model == True:
        cfg.save_model_dirname = "./save_models"
        cfg.save_model_filename = "Model"

    """ Running """
    rs.run_serial(cfg, model, loss_fn, train_datasets, test_dataset, args.dataset)


if __name__ == "__main__":
    main()

# To run:
# python ./mnist_no_mpi.py
