import pytest
import os
import numpy as np
import torch

from appfl.config import *
from appfl.misc.data import *
from appfl.misc.utils import *

import appfl.run_serial as rs

import argparse

import scipy 


class LinearRegression(torch.nn.Module):
    def __init__(self, inputsize, outputsize):
        super(LinearRegression, self).__init__()
        self.fc1 = torch.nn.Linear(inputsize, outputsize, bias=False)

    def forward(self, x):
        out = self.fc1(x)
        return out

def get_data(cfg): 
    currentpath = os.getcwd()    
    mat_path = currentpath + "/examples/datasets/RawData/fedrecon/xrf_xrt_toy.mat"  

    data = scipy.io.loadmat(mat_path)
    A = torch.from_numpy(data["A"].toarray()).to(torch.float32)
    b1 = torch.from_numpy(data["b1"]).to(torch.float32)
    b2 = torch.from_numpy(data["b2"]).to(torch.float32)
    b3 = torch.from_numpy(data["b3"]).to(torch.float32)
    b = torch.from_numpy(data["b"]).to(torch.float32)

    ## coefficient assigned for each client    
    c = torch.from_numpy(data["c"][0]).to(torch.float32)
    for idx in range(len(c)):
        cfg.fed.args.coeff[idx] = c[idx].item()
         
    ## ground truth (to compute the mean squared error for every iterations)
    w1_truth = data["w1"].transpose().reshape(-1).tolist()
    w2_truth = data["w2"].transpose().reshape(-1).tolist()
    w3_truth = data["w3"].transpose().reshape(-1).tolist()
    w_truth = data["w"].transpose().reshape(-1).tolist()

    for idx in range(len(w1_truth)):
        cfg.fed.args.w_truth[0].append(w1_truth[idx])
        cfg.fed.args.w_truth[1].append(w2_truth[idx])
        cfg.fed.args.w_truth[2].append(w3_truth[idx])
        cfg.fed.args.w_truth[3].append(w_truth[idx])

    train_datasets = []
    train_datasets.append(Dataset(A, b1))
    train_datasets.append(Dataset(A, b2))
    train_datasets.append(Dataset(A, b3))
    train_datasets.append(Dataset(A, b))

    return train_datasets
 

def test_aps_fedres():
    
    cfg = OmegaConf.structured(Config)    
    cfg.fed = eval("Fedres()")        
    cfg.train_data_shuffle=False    
    cfg.save_model_state_dict = False
    set_seed(1)
    cfg.num_clients = 4
    cfg.fed.args.optim = "SGD"
    cfg.fed.args.optim_args.lr = 1e3
    cfg.fed.args.num_local_epochs = 1
    cfg.batch_training = False
    cfg.train_data_batch_size = 354

    """ User-defined data """
    train_datasets = get_data(cfg)

    """ User-defined model """
    num_feature = train_datasets[0][0][0].size(dim=0)
    model = LinearRegression(num_feature, 1)
    loss_fn = torch.nn.MSELoss(reduction="sum")

    rs.run_serial(cfg, model, loss_fn, train_datasets, Dataset(), "test_aps", None)
 