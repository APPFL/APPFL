import sys
import time
import logging
import argparse
import math
import json 

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import ToTensor

from appfl.config import *
from appfl.misc.data import *
import appfl.run_grpc_client as grpc_client


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
    parser.add_argument("--host", type=str, required=True)
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--use_tls", type=bool, default=False)
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--nclients", type=int, required=True)
    parser.add_argument("--logging", type=str, default="INFO")
    parser.add_argument("--api_key", default=None)
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=eval("logging." + args.logging))
    

    start_time = time.time()
    # """ CNN """    
    # model = CNN(3, 2, 32)
    # cfg.fed.args.loss_type = "torch.nn.CrossEntropyLoss()"
     
    # with open("../datasets/PreprocessedData/deepcovid_test_data.json") as f:
    #     train_data_raw = json.load(f)
    
    # train_dataset = Dataset(
    #         torch.FloatTensor(train_data_raw["x"]), torch.tensor(train_data_raw["y"])
    #     )
     
    
    """ Isabelle's DenseNet (the outputs of the model are probabilities of 1 class ) """
    import importlib.machinery
    loader = importlib.machinery.SourceFileLoader('MainModel', './IsabelleTorch.py')
    MainModel = loader.load_module()
 

    file = "./IsabelleTorch.pth"         
    model = torch.load(file)        
    model.eval()
    cfg.fed.args.loss_type = "torch.nn.BCELoss()"

    with open("../datasets/PreprocessedData/deepcovid32_test_data.json") as f:
        train_data_raw = json.load(f)
    
    train_dataset = Dataset(
            torch.FloatTensor(train_data_raw["x"]), torch.FloatTensor(train_data_raw["y"]).reshape(-1,1)
        )

    logger = logging.getLogger(__name__)
    logger.info(f"----------Loaded Datasets and Model----------Elapsed Time={time.time() - start_time}")

    
    cfg.server.host = args.host
    cfg.server.port = args.port
    cfg.server.use_tls = args.use_tls
    cfg.server.api_key = args.api_key
    logger.debug(OmegaConf.to_yaml(cfg))

    grpc_client.run_client(
        cfg, args.client_id, model, train_dataset
    )
    logger.info("------DONE------")


if __name__ == "__main__":
    main()
