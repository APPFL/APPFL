import os
import time

import numpy as np
import torch

import torchvision
from torchvision.transforms import ToTensor

from appfl.config import *
from appfl.misc.data import *
from appfl.misc.utils import *
from models.resnet_fda import *
from datasets.MIDRC import *

import appfl.run_serial as rs

import argparse

""" read arguments """

parser = argparse.ArgumentParser()

parser.add_argument("--device", type=str, default="cpu")

## dataset
parser.add_argument("--dataset", type=str, default="MIDRC")
# parser.add_argument("--num_channel", type=int, default=1)
# parser.add_argument("--num_classes", type=int, default=10)
# parser.add_argument("--num_pixel", type=int, default=28)
parser.add_argument("--base_data_path", type=str, default='/u/enyij2/data/midrc')
parser.add_argument('--data_aug_times', type=int, default=1)
parser.add_argument('--target', type=int, default=0, help='target domain idx')
parser.add_argument('--n_target_samples', type=int, default=2000)
parser.add_argument('--source_batch_size', type=int, default=8)
parser.add_argument('--target_batch_size', type=int, default=16)

## model
parser.add_argument("--hidden_size", type=int, default=128)
parser.add_argument('--resnet', type=str, default='resnet18')

## clients
parser.add_argument("--num_clients", type=int, default=5)
parser.add_argument("--client_optimizer", type=str, default="Adam")
parser.add_argument("--client_lr", type=float, default=1e-4)
parser.add_argument("--num_local_epochs", type=int, default=1)

## server
parser.add_argument("--server", type=str, default="ServerFedAvg")
parser.add_argument("--num_epochs", type=int, default=20)

parser.add_argument("--server_lr", type=float, required=False)
parser.add_argument("--mparam_1", type=float, required=False)
parser.add_argument("--mparam_2", type=float, required=False)
parser.add_argument("--adapt_param", type=float, required=False)


args = parser.parse_args()

if torch.cuda.is_available():
    args.device = "cuda"
 
# rewrite it for MIDRC
def get_data(target, states, transform):
    dls = {'train':[], 'test':[]}
    for mode in ['train', 'test']:
        for i in range(args.num_clients):
            if i != target and mode == 'test':
                continue
            elif i == target and mode == 'train':
                dataset = MidrcDataset(os.path.join(args.base_data_path, 'meta_info', f'MIDRC_table_{states[i]}_{mode}.csv'), base_path=args.base_data_path, augment_times=args.data_aug_times, transform=transform[mode], n_samples=args.n_target_samples)
            else:
                dataset = MidrcDataset(os.path.join(args.base_data_path, 'meta_info', f'MIDRC_table_{states[i]}_{mode}.csv'), base_path=args.base_data_path, augment_times=args.data_aug_times, transform=transform[mode])
            dls[mode].append(dataset)
    return dls['train'], dls['test'][0]

# rewrite it for FDA
def get_model():
    ## User-defined model
    model = ResNetClassifier(resnet=args.resnet, hidden_size=args.hidden_size)
    return model


## Run
def main():

    """ Configuration """
    cfg = OmegaConf.structured(Config)

    cfg.device = args.device
    cfg.reproduce = True
    if cfg.reproduce == True:
        set_seed(1)

    ## clients
    cfg.num_clients = args.num_clients
    cfg.fed.args.optim = args.client_optimizer
    cfg.fed.args.optim_args.lr = args.client_lr
    cfg.fed.args.num_local_epochs = args.num_local_epochs

    ## server
    cfg.fed.servername = args.server
    cfg.num_epochs = args.num_epochs
    
    ## datasets
    cfg.train_data_batch_size = args.source_batch_size
    cfg.test_data_batch_size = args.target_batch_size

    ## outputs

    cfg.use_tensorboard = False

    cfg.save_model_state_dict = False

    cfg.output_dirname = "./outputs_%s_%s_%s" % (
        args.dataset,
        args.server,
        args.client_optimizer,
    )
    if args.server_lr != None:
        cfg.fed.args.server_learning_rate = args.server_lr
        cfg.output_dirname += "_ServerLR_%s" % (args.server_lr)

    if args.adapt_param != None:
        cfg.fed.args.server_adapt_param = args.adapt_param
        cfg.output_dirname += "_AdaptParam_%s" % (args.adapt_param)

    if args.mparam_1 != None:
        cfg.fed.args.server_momentum_param_1 = args.mparam_1
        cfg.output_dirname += "_MParam1_%s" % (args.mparam_1)

    if args.mparam_2 != None:
        cfg.fed.args.server_momentum_param_2 = args.mparam_2
        cfg.output_dirname += "_MParam2_%s" % (args.mparam_2)

    cfg.output_filename = "result"

    start_time = time.time()

    """ User-defined model """
    model = get_model()
    loss_fn = torch.nn.CrossEntropyLoss()   
    # loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')

    ## loading models
    cfg.load_model = False
    if cfg.load_model == True:
        cfg.load_model_dirname = "./save_models"
        cfg.load_model_filename = "Model"
        model = load_model(cfg)

    """ User-defined data """
    transform = {
        'train':
        transforms.Compose([ 
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                 inplace=True),
            ]),
        'test':
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                 inplace=True),
            ])
    }
    
    states = ['IL', 'NC', 'CA', 'IN', 'TX']
    train_datasets, test_dataset = get_data(args.target, states, transform)

    ## Sanity check for the user-defined data
    # if cfg.data_sanity == True:
    #     data_sanity_check(
    #         train_datasets, test_dataset, args.num_channel, args.num_pixel
    #     )

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
    if args.server == 'ServerFedAvg':
        train_datasets.remove(train_datasets[args.target])
        cfg.num_clients -= 1
        rs.run_serial(cfg, model, loss_fn, train_datasets, test_dataset, args.dataset)
        


if __name__ == "__main__":
    main()

# To run:
# python ./mnist_no_mpi.py
