
import sys

sys.path.insert(0, "..")

import time

start_time = time.time()

## User-defined datasets
import numpy as np
import torch
from appfl.misc.data import *
import glob
import cv2
import csv

DataSet_name = "Coronahack"
num_clients = 4
num_channel = 3    # 1 if gray, 3 if color
num_classes = 7   # number of the image classes
num_pixel = 32   # image size = (num_pixel, num_pixel)

dir = "./datasets/RawData/%s/archive" % (DataSet_name)


class Coronahack():
    def __init__(self, dir, pixel, is_train): 

        if is_train == True:        
            self.imgs_path = dir+"/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/"
        else:
            self.imgs_path = dir+"/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/"

        self.data = []
        with open(dir+"/Chest_xray_Corona_Metadata.csv", 'r') as file:
            csvreader = csv.reader(file)
            header = next(csvreader)
            for row in csvreader:     
                if is_train == True:      
                    if row[3]=="TRAIN":
                        img_path = self.imgs_path + row[1]
                        class_name = row[2]+row[4]+row[5]
                        self.data.append([img_path, class_name])     
                else:                                          
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
        
        self.img_dim   = (pixel, pixel)  

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]         
        img = cv2.imread(img_path)             
        img = cv2.resize(img, self.img_dim)        
        class_id = self.class_map[class_name]        
        img_tensor = torch.from_numpy(img) / 256
        img_tensor = img_tensor.permute(2, 0, 1)              
        
        return img_tensor, class_id

# test data for a server
test_data_raw=Coronahack(dir, num_pixel, is_train=False)		

test_data_input = []
test_data_label = []
for idx in range(len(test_data_raw)):    
    test_data_input.append( test_data_raw[idx][0].tolist() )
    test_data_label.append( test_data_raw[idx][1] )

test_dataset = Dataset(
    torch.FloatTensor(test_data_input), torch.tensor(test_data_label)
)


# training data for multiple clients
train_data_raw=Coronahack(dir, num_pixel, is_train=True)		
split_train_data_raw = np.array_split(range(len(train_data_raw)), num_clients)     
train_datasets=[]
for i in range(num_clients):
    train_data_input = []
    train_data_label = []
    for idx in split_train_data_raw[i]:        
        train_data_input.append(train_data_raw[idx][0].tolist())
        train_data_label.append(train_data_raw[idx][1])
        
    train_datasets.append(
            Dataset(
                torch.FloatTensor(train_data_input),
                torch.tensor(train_data_label),
            )
        )


data_sanity_check(train_datasets, test_dataset, num_channel, num_pixel)

## User-defined model
from examples.models.cnn import *

model = CNN(num_channel, num_classes, num_pixel)

print(
    "----------Loaded Datasets and Model----------Elapsed Time=",
    time.time() - start_time,
)

## Run
import appfl.run as rt
import hydra
from mpi4py import MPI
from omegaconf import DictConfig


@hydra.main(config_path="../appfl/config", config_name="config")
def main(cfg: DictConfig):

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    torch.manual_seed(1)

    if comm_size > 1:
        if comm_rank == 0:
            rt.run_server(cfg, comm, model, test_dataset, num_clients, DataSet_name)
        else:
            rt.run_client(cfg, comm, model, train_datasets, num_clients)
        print("------DONE------", comm_rank)
    else:
        rt.run_serial(cfg, model, train_datasets, test_dataset, DataSet_name)


if __name__ == "__main__":
    main()
 

# To run CUDA-aware MPI:
# mpiexec -np 5 --mca opal_cuda_support 1 python ./coronahack.py
# To run MPI:
# mpiexec -np 5 python ./coronahack.py
# To run:
# python ./coronahack.py
