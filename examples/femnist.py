import sys

sys.path.append("..")

## User-defined datasets
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

num_channel = 1    # 1 if gray, 3 if color
num_classes = 62   # number of the image classes 
num_pixel   = 28   # image size = (num_pixel, num_pixel)

# (1) Train Datasets for every clients
class FEMNIST_Train(Dataset):
    def __init__(self, train_data_image, train_data_class): 
        self.train_data_image = train_data_image
        self.train_data_class = train_data_class                
    def __len__(self):
        return len(self.train_data_class)
    def __getitem__(self, idx):
        return self.train_data_image[idx], self.train_data_class[idx]

train_data_raw={}  
train_data_image={}  
train_data_class={}  
for idx in range(36):            
    with open("../datasets/FEMNIST/train/all_data_%s_niid_05_keep_0_train_9.json"%(idx)) as f:    
        train_data_raw[idx] = json.load(f)    
    for client in train_data_raw[idx]["users"]:    
        train_data_image[client] = []        
        
        for image_data in train_data_raw[idx]["user_data"][client]["x"]:                    
            image_data = np.asarray(image_data)
            image_data.resize(28,28)   
            train_data_image[client].append([image_data])
                        
        train_data_image[client] = torch.FloatTensor(train_data_image[client])
        train_data_class[client] = torch.tensor(train_data_raw[idx]["user_data"][client]["y"])
  
train_datasets=[]
for client in train_data_image.keys():
    train_datasets.append(FEMNIST_Train(train_data_image[client], train_data_class[client]))

# (2) Test Dataset for the server
class FEMNIST_Test(Dataset):
    def __init__(self): 
        
        test_data_raw={}  
        test_data_image=[] 
        test_data_class=[] 
        for idx in range(36):            
            with open("../datasets/FEMNIST/test/all_data_%s_niid_05_keep_0_test_9.json"%(idx)) as f:    
                test_data_raw[idx] = json.load(f)    
            for client in test_data_raw[idx]["users"]:                                
                for image_data in test_data_raw[idx]["user_data"][client]["x"]:                    
                    image_data = np.asarray(image_data)
                    image_data.resize(28,28)   
                    test_data_image.append([image_data])

                for class_data in test_data_raw[idx]["user_data"][client]["y"]:                          
                    test_data_class.append(class_data)


        self.test_data_image = torch.FloatTensor(test_data_image)
        self.test_data_class = torch.tensor(test_data_class)     

    def __len__(self):
        return len(self.test_data_class)
    def __getitem__(self, idx):
        return self.test_data_image[idx], self.test_data_class[idx]

test_dataset = FEMNIST_Test()
 

# (3) Check if "DataLoader" from PyTorch works.
train_dataloader = DataLoader(train_datasets[0], batch_size=64, shuffle=False)    
for image, class_id in train_dataloader:
    assert(image.shape[0] == class_id.shape[0])
    assert(image.shape[1] == num_channel)
    assert(image.shape[2] == num_pixel)
    assert(image.shape[3] == num_pixel)

test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)    
for image, class_id in test_dataloader:
    assert(image.shape[0] == class_id.shape[0])
    assert(image.shape[1] == num_channel)
    assert(image.shape[2] == num_pixel)
    assert(image.shape[3] == num_pixel) 


## User-defined model

import torch.nn as nn
import math


class CNN1(nn.Module):
    def __init__(self, in_features, num_classes, pixel):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_features, 32, kernel_size=5, padding=0, stride=1, bias=True
        )
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        self.act = nn.ReLU(inplace=True)

        ###
        ### X_out = floor{ 1 + (X_in + 2*padding - dilation*(kernel_size-1) - 1)/stride }
        ###
        X = pixel
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


model = CNN1(num_channel, num_classes, num_pixel)

print("----------Loaded Datasets and Model----------")

## train
import appfl.run_trial as rt
import hydra
from omegaconf import DictConfig
from mpi4py import MPI

@hydra.main(config_path="../appfl/config", config_name="config")
def main(cfg: DictConfig):
    
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
 
    torch.manual_seed(1)
 
    if comm_size > 1:
        if comm_rank == 0:
            rt.run_server(cfg, comm, model, test_dataset)
        else:
            rt.run_client(cfg, comm, model, train_datasets)

        print("------DONE------", comm_rank)
    else:
        rt.run_serial(cfg, model, train_datasets, test_dataset)
        

if __name__ == "__main__":
    main()


# To run CUDA-aware MPI:
# mpiexec -np 5 --mca opal_cuda_support 1 python ./run.py
# To run MPI:
# mpiexec -np 5 python ./femnist.py
# To run:
# python ./femnist.py
