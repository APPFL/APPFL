import sys

sys.path.append("..")

## User-defined datasets
import csv
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

DataSet_name = "Coronahack" 
num_channel = 3    # 1 if gray, 3 if color
num_classes = 7    # number of the image classes 
num_pixel   = 32   # image size = (num_pixel, num_pixel)
num_clients = 4

# (1) Train Datasets for every clients
class Coronahack_Train_Raw(Dataset):
    def __init__(self, num_channel, num_classes, num_pixel):         
        self.imgs_path = "../datasets/Coronahack/archive/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/"        
        self.data = []
        with open("../datasets/Coronahack/archive/Chest_xray_Corona_Metadata.csv", 'r') as file:
            csvreader = csv.reader(file)
            header = next(csvreader)
            for row in csvreader:
                if row[3]=="TRAIN":                                   
                    img_path = self.imgs_path + row[1]
                    class_name = row[2]+row[4]+row[5]
                    self.data.append([img_path, class_name])            

        class_name_list =[]                     
        for img_path, class_name in self.data:
            if class_name not in class_name_list:
                class_name_list.append(class_name)

        self.class_map = {}
        tmpcnt = 0
        for class_name in class_name_list:
            self.class_map[class_name] = tmpcnt
            tmpcnt += 1  

        self.img_dim   = (num_pixel, num_pixel)  


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]        
        img = cv2.imread(img_path)        
        img = cv2.resize(img, self.img_dim)        
        class_id = self.class_map[class_name]        
        img_tensor = torch.from_numpy(img) / 256
        img_tensor = img_tensor.permute(2, 0, 1)
        class_id = torch.tensor(class_id)
        return img_tensor, class_id

class Coronahack_Train(Dataset):
    def __init__(self, train_data_image, train_data_class): 
        self.train_data_image = train_data_image
        self.train_data_class = train_data_class                
    def __len__(self):
        return len(self.train_data_class)
    def __getitem__(self, idx):
        return self.train_data_image[idx], self.train_data_class[idx]



train_data_raw = Coronahack_Train_Raw(num_channel, num_classes, num_pixel)
train_data_image={}  
train_data_class={}  


split_train_data_raw= np.array_split(range(len(train_data_raw)), num_clients) 

for i in range(num_clients):
    train_data_image[i] = []
    train_data_class[i] = []
    for idx in split_train_data_raw[i]:        
        train_data_image[i].append(train_data_raw[idx][0])
        train_data_class[i].append(train_data_raw[idx][1].item())
    
    train_data_class[i] = torch.tensor(train_data_class[i])

train_datasets=[]; 
for client in train_data_image.keys():
    train_datasets.append(Coronahack_Train(train_data_image[client], train_data_class[client]))
    
num_clients = len(train_datasets)        

# (2) Test Dataset for the server
class Coronahack_Test(Dataset):
    def __init__(self, num_channel, num_classes, num_pixel): 
        
        self.imgs_path = "../datasets/Coronahack/archive/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/"
        self.data = []
        with open("../datasets/Coronahack/archive/Chest_xray_Corona_Metadata.csv", 'r') as file:
            csvreader = csv.reader(file)
            header = next(csvreader)
            for row in csvreader:                
                if row[3]=="TEST":                    
                    img_path = self.imgs_path + row[1]
                    class_name = row[2]+row[4]+row[5]
                    self.data.append([img_path, class_name])            

        class_name_list =[]                     
        for img_path, class_name in self.data:
            if class_name not in class_name_list:
                class_name_list.append(class_name)

        self.class_map = {}
        tmpcnt = 0
        for class_name in class_name_list:
            self.class_map[class_name] = tmpcnt
            tmpcnt += 1 
        
        self.img_dim = (num_pixel, num_pixel)  

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]         
        img = cv2.imread(img_path)                
        img = cv2.resize(img, self.img_dim)        
        class_id = self.class_map[class_name]        
        img_tensor = torch.from_numpy(img) / 256
        img_tensor = img_tensor.permute(2, 0, 1)
        class_id = torch.tensor(class_id)
        return img_tensor, class_id

test_dataset = Coronahack_Test(num_channel, num_classes, num_pixel)
 

# (3) Check if "DataLoader" from PyTorch works.
train_dataloader = DataLoader(train_datasets[0], batch_size=64, shuffle=False)    
for image, class_id in train_dataloader:    
    print(image.shape, "  ", class_id)
#     assert(image.shape[0] == class_id.shape[0])
#     assert(image.shape[1] == num_channel)
#     assert(image.shape[2] == num_pixel)
#     assert(image.shape[3] == num_pixel)

# test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)    
# for image, class_id in test_dataloader:
#     print(image.shape, "  ", class_id)
#     assert(image.shape[0] == class_id.shape[0])
#     assert(image.shape[1] == num_channel)
#     assert(image.shape[2] == num_pixel)
#     assert(image.shape[3] == num_pixel) 


## User-defined model

import torch.nn as nn
import math


class CNN(nn.Module):
    def __init__(self, num_channel, num_classes, num_pixel):
        super().__init__()
        self.conv1 = nn.Conv2d(
            num_channel, 32, kernel_size=5, padding=0, stride=1, bias=True
        )
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        self.act = nn.ReLU(inplace=True)

        ###
        ### X_out = floor{ 1 + (X_in + 2*padding - dilation*(kernel_size-1) - 1)/stride }
        ###
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


model = CNN(num_channel, num_classes, num_pixel)

print("----------Loaded Datasets and Model----------")
print("----------Check train_datasets, test_dataset, num_clients----------")

## train
# import appfl.run_trial as rt
# import hydra
# from omegaconf import DictConfig
# from mpi4py import MPI

# @hydra.main(config_path="../appfl/config", config_name="config")
# def main(cfg: DictConfig):
    
#     comm = MPI.COMM_WORLD
#     comm_rank = comm.Get_rank()
#     comm_size = comm.Get_size()
 
#     torch.manual_seed(1)
 
#     if comm_size > 1:
#         if comm_rank == 0:
            
#             rt.run_server(cfg, comm, model, test_dataset, num_clients, DataSet_name)

#         else:            
#             num_client_groups = np.array_split(range(num_clients), comm_size - 1)            
            
#             clients_dataloaders=[]
#             for _, cid in enumerate(num_client_groups[comm_rank - 1]):

#                 ## TO DO: advance techniques (e.g., utilizing batch)
#                 if cfg.fed.type == "iadmm":  
#                     cfg.batch_size = len(train_datasets[cid])
                
#                 clients_dataloaders.append(
#                     DataLoader( train_datasets[cid], num_workers=0, batch_size=cfg.batch_size, shuffle=False)
#                 )
            
#             rt.run_client(cfg, comm, model, clients_dataloaders, num_client_groups)

#         print("------DONE------", comm_rank)
#     else:

#         rt.run_serial(cfg, model, train_datasets, test_dataset)
        

# if __name__ == "__main__":
#     main()


# To run CUDA-aware MPI:
# mpiexec -np 5 --mca opal_cuda_support 1 python ./coronahack.py
# To run MPI:
# mpiexec -np 5 python ./coronahack.py
# To run:
# python ./coronahack.py
 

 