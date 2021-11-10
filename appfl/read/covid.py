import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import tarfile
import csv

class CovidTrain(Dataset):
    def __init__(self, in_features, num_classes, pixel): 
        dir = "../../../datasets"

        self.imgs_path = dir+"/archive/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/"        
        self.data = []
        with open(dir+"/archive/Chest_xray_Corona_Metadata.csv", 'r') as file:
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
        class_id = torch.tensor(class_id)
        return img_tensor, class_id

class CovidTest(Dataset):
    def __init__(self, in_features, num_classes, pixel): 
        dir = "../../../datasets"
        self.imgs_path = dir+"/archive/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/"
        self.data = []
        with open(dir+"/archive/Chest_xray_Corona_Metadata.csv", 'r') as file:
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
        class_id = torch.tensor(class_id)
        return img_tensor, class_id
 
	
		
		