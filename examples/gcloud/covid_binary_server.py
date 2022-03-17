import sys
import time
import logging
import argparse
import math
import json

import torch
import torch.nn as nn

from appfl.config import *
import appfl.run_grpc_server as grpc_server

from appfl.misc.data import *
 


class CNN(nn.Module):
    def __init__(self, num_channel, num_classes, num_pixel):
        super().__init__()
        self.conv1 = nn.Conv2d(
            num_channel, 32, kernel_size=5, padding=0, stride=1, bias=True
        )
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        self.act = nn.ReLU(inplace=True)

        X = num_pixel
        X = math.floor(1 + (X + 2 * 0 - 1 * (5 - 1) - 1) / 1)
        X = X / 2
        X = math.floor(1 + (X + 2 * 0 - 1 * (5 - 1) - 1) / 1)
        X = X / 2
        X = int(X)

        self.fc1 = nn.Linear(64 * X * X, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.maxpool(x)
        x = self.act(self.conv2(x))
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    # read default configuration
    cfg = OmegaConf.structured(Config)

    parser = argparse.ArgumentParser(description="Provide the configuration")
    
    parser.add_argument("--nclients", type=int, required=True)
    parser.add_argument("--total_iter", type=int, required=True)
    parser.add_argument("--local_iter", type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)      
    parser.add_argument("--check_intv", type=int, required=True)
    
    parser.add_argument("--logging", type=str, default="INFO")
    args = parser.parse_args()

    cfg.num_epochs = args.total_iter
    cfg.fed.args.num_local_epochs = args.local_iter
    cfg.fed.args.optim_args.lr = args.lr

    logging.basicConfig(stream=sys.stdout, level=eval("logging." + args.logging))
    
    start_time = time.time()    

    # """ CNN """    
    # model = CNN(3, 2, 32)
    
    """ Isabelle's DenseNet (the outputs of the model are probabilities of 1 class ) """
    import importlib.machinery
    loader = importlib.machinery.SourceFileLoader('MainModel', './IsabelleTorch.py')
    MainModel = loader.load_module()
 

    file = "./IsabelleTorch.pth"         
    model = torch.load(file)        
    model.eval()
    cfg.fed.args.loss_type = "torch.nn.BCELoss()" 

    logger = logging.getLogger(__name__)
    logger.info(f"----------Loaded Data and Model----------Elapsed Time={time.time() - start_time}")
    logger.debug(OmegaConf.to_yaml(cfg))

    """ saving models """
    cfg.save_model = True
    if cfg.save_model == True:
        cfg.save_model_dirname      = "./save_models"
        cfg.save_model_filename     = "Covid_Binary_Isabelle_FedAvg" 
        cfg.checkpoints_interval    = args.check_intv

    grpc_server.run_server(cfg, model, args.nclients)


if __name__ == "__main__":
    main()
