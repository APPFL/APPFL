import sys

from models.cnn import *
from models.resnet import *
from models.utils import *

import argparse
import copy 
import torch
import torch.optim as optim

import time
import os
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import logging
  
import numpy as np

from appfl.config import *
from appfl.misc.data import *
from appfl.misc.utils import *
import appfl.run as rt

from mpi4py import MPI

""" read arguments """
 
parser = argparse.ArgumentParser()

parser.add_argument('--optimizer', type=str, required=True)   
parser.add_argument('--num_epochs', type=int, required=True)  
parser.add_argument('--LR', type=float, required=True)  

parser.add_argument("--num_clients", type=int, default=1)
parser.add_argument("--num_pixel", type=int, default=32)
parser.add_argument('--num_workers', type=int, default=32)  

args = parser.parse_args()    

args.device = "cuda" 
args.dataset_name = "CIFAR10"
args.num_classes = 10
args.loss_fn = "torch.nn.CrossEntropyLoss()"              

""" output """
name="%s_%s_LR_%s" %(args.dataset_name, args.optimizer,args.LR)

args.output_dirname = "./%s/" %(name)
args.output_filename = "summary"

logger = logging.getLogger(__name__)
logger = create_custom_logger(logger, args)


def get_data(args):

    dir = os.getcwd() + "/datasets/RawData"

    # train_dataset = eval("torchvision.datasets." + args.dataset_name)(
    #         dir, download=True, train=True, 
    #         transform=transforms.Compose([                        
    #                     transforms.ToTensor(),                        
    #                 ])
    # )
 

    # test_dataset = eval("torchvision.datasets." + args.dataset_name)(
    #     dir, download=True, train=False,  transform=transforms.Compose([
    #                 transforms.ToTensor(),                    
    #             ])
    # ) 
  
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = eval("torchvision.datasets." + args.dataset_name)(
            dir, download=True, train=True, 
            transform=transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, 4),
                        transforms.ToTensor(),
                        normalize,
                    ])
        )
 

    test_dataset = eval("torchvision.datasets." + args.dataset_name)(
        dir, download=True, train=False,  transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ])
    ) 

    split_train_data_raw = np.array_split(range(len(train_dataset)), args.num_clients)        
    train_datasets = []
    for i in range(args.num_clients):
        train_datasets.append(torch.utils.data.Subset(train_dataset, split_train_data_raw[i]))
            
    return train_datasets, test_dataset

def main():
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
  
    # """ Configuration """     
    cfg = OmegaConf.structured(Config)     
    cfg.device = args.device      
    cfg.output_filename += "_cifar10_%s_LR_%s_npixel_%s" %(args.optimizer, args.LR, args.num_pixel)


    start_time = time.time()
    """ data """ 
    train_datasets, test_dataset = get_data(args)

    """ model """ 
    model = resnet20(num_classes=args.num_classes)  
    # model = CNN(3, 10, 32)  
    
    cfg.fed.args.loss_type = args.loss_fn

    """ Optimizer """
    cfg.fed.servername = "ServerFedAvg"
    cfg.num_epochs = args.num_epochs    

    cfg.fed.args.optim = args.optimizer
    cfg.fed.args.num_local_epochs = 3        
    cfg.fed.args.optim_args.lr = args.LR
     
    loading_time = round( time.time() - start_time, 2)
    logger.info("Loading Time = %s" %(loading_time))

    # """ Running """     
    if comm_size > 1:
        if comm_rank == 0:
            rt.run_server(cfg, comm, model, args.num_clients, test_dataset, args.dataset_name)
        else:
            rt.run_client(cfg, comm, model, args.num_clients, train_datasets)
        print("------DONE------", comm_rank)
    else:
        rt.run_serial(cfg, model, train_datasets, test_dataset, args.dataset_name)
   
    # """ Local Training """ 
    # if args.optimizer=="SGD":
    #     optimizer = optim.SGD(model.parameters(), lr=args.LR, momentum=0.9, weight_decay=1e-4)     
    
    # if args.optimizer=="Adam":
    #     optimizer = optim.Adam(model.parameters(), lr=args.LR, weight_decay=0)

    # train_dataloader = torch.utils.data.DataLoader(
    #                 train_datasets[0],
    #                 batch_size=64, shuffle=True,
    #                 num_workers=32, pin_memory=True)
    # test_dataloader = DataLoader(
    #             test_dataset,
    #             num_workers=32,
    #             batch_size=64,
    #             shuffle=False,
    #         )     
                                
    # """ Initial Parameter """
    # valid_stime = time.time()            
    # train_loss, train_accuracy, train_auc = local_validation(args, copy.deepcopy(model), train_dataloader)
    # test_loss, test_accuracy, test_auc = local_validation(args, copy.deepcopy(model), test_dataloader)
    # valid_etime = time.time() - valid_stime
    # logging_initial(logger, -1, 0.0, valid_etime, 0.0, train_loss, train_accuracy, train_auc, test_loss, test_accuracy, test_auc)  
    
    # start_time = time.time()
    # model.train()    
    # for t in range(args.num_epochs):
    #     train_stime = time.time()
    #     for data, target in train_dataloader:
    #         data = data.to(args.device)
    #         target = target.to(args.device)
    #         optimizer.zero_grad()
    #         output = model(data)
    #         loss = eval(args.loss_fn)(output, target)
    #         loss.backward() 
    #         optimizer.step()
 
    #     train_etime = round(time.time() - train_stime,2)

    #     valid_stime = time.time()            
    #     train_loss, train_accuracy, train_auc = local_validation(args, copy.deepcopy(model), train_dataloader)            
    #     test_loss, test_accuracy, test_auc = local_validation(args, copy.deepcopy(model),  test_dataloader)
    #     valid_etime = time.time() - valid_stime

    #     elapsed_time = round(time.time()-start_time,2)

    #     logging_iteration(logger, t, train_etime, valid_etime, elapsed_time, train_loss, train_accuracy, train_auc, test_loss, test_accuracy, test_auc)


if __name__ == "__main__":
    main()            