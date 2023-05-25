import os
import time

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
    
""" read arguments """

parser = argparse.ArgumentParser()

parser.add_argument("--device", type=str, default="cpu")

## dataset
parser.add_argument("--dataset", type=str, default="xrf_xrt_noisy_scenario_2")

## algorithm
parser.add_argument("--federation_type", type=str, default="Fedres")   

## server  
parser.add_argument("--num_epochs", type=int, default=2)
  
## clients
parser.add_argument("--num_clients", type=int, default=4) ## FIX: {0: XRF0, 1: XRF1, 2: XRF2, 3:XRT} 
parser.add_argument("--client_optimizer", type=str, default="SGD")
parser.add_argument("--client_lr", type=float, default=1e3)
parser.add_argument("--num_local_epochs", type=int, default=1)
parser.add_argument("--batch_training", type=bool, default=False)
parser.add_argument("--train_data_batch_size", type=int, default=354)


args = parser.parse_args()

if torch.cuda.is_available():
    args.device = "cuda"
 

def get_data(cfg):        
    """ Regarding the training dataset: "xrf_xrt_noisy_scenario_2" as an example
    client_0 (XRF): matrix "A" and vector "b1"
    client_1 (XRF): matrix "A" and vector "b2"
    client_2 (XRF): matrix "A" and vector "b3"
    client_3 (XRT): matrix "A" and vector "b"
    - matrix A has 35400 rows and 62500 columns 
    - vector b (as well as b1, b2, and b3) has 35400 elements
    - each row of A represents a data sample at a specific (angle, beamlet). In this dataset, we consider 100 different angles and 354 beamlets. The first 354 rows have the same angle in common, and so on.
    - each column of A is associated with pixel. In this example, there are 250*250 pixels.
    """

    mat_path = os.getcwd() + "/datasets/RawData/fedrecon/%s.mat"%(args.dataset)
 
    data = scipy.io.loadmat(mat_path)       
    A = torch.from_numpy(data['A'].toarray()).to(torch.float32)        
    b1 = torch.from_numpy(data['b1']).to(torch.float32)     
    b2 = torch.from_numpy(data['b2']).to(torch.float32)     
    b3 = torch.from_numpy(data['b3']).to(torch.float32)     
    b = torch.from_numpy(data['b']).to(torch.float32)    
    
    ## coefficient assigned for each client
    c = torch.from_numpy(data['c'][0]).to(torch.float32)     
    for idx in range(len(c)):
        cfg.fed.args.coeff[idx]=c[idx].item()
    ## ground truth (to compute the mean squared error for every iterations)    
    w1_truth = data['w1'].tolist()
    w2_truth = data['w2'].tolist()
    w3_truth = data['w3'].tolist()
    w_truth = data['w'].tolist()
 
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

## Run
def main():

    """ Configuration """
    cfg = OmegaConf.structured(Config)

    cfg.device = args.device
    cfg.reproduce = True
    if cfg.reproduce == True:
        set_seed(1)
    
    ## Turn off save models for all client
    cfg.save_model_state_dict = False
    
    ## outputs  
    cfg.output_dirname = "./outputs_%s" % (args.dataset) 
    cfg.output_filename = "result"
         
    ## algorithm
    cfg.fed = eval(args.federation_type+"()")    
    
    ## server  
    cfg.num_epochs = args.num_epochs
     
    ## clients
    cfg.num_clients = args.num_clients
    cfg.fed.args.optim = args.client_optimizer
    cfg.fed.args.optim_args.lr = args.client_lr
    cfg.fed.args.num_local_epochs = args.num_local_epochs
    cfg.batch_training = args.batch_training
    cfg.train_data_batch_size = args.train_data_batch_size
    
    """ User-defined data """
    train_datasets = get_data(cfg)
    
    """ User-defined model """
    num_feature = train_datasets[0][0][0].size(dim=0)
    model = LinearRegression(num_feature, 1)
    loss_fn = torch.nn.MSELoss(reduction='sum')

    """ Running """
    rs.run_serial(cfg, model, loss_fn, train_datasets, Dataset(), args.dataset)
        

if __name__ == "__main__":
    main()

# To run:
# python ./fedrecon_no_mpi.py
