import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import csv

class CustomDataset(Dataset):
    def __init__(self): 
        self.imgs_path = "./datasets/Corona/Coronahack-Chest-XRay-Dataset/train/"
        
        # rows  = []
        self.data = []
        with open("./datasets/Corona/Chest_xray_Corona_Metadata.csv", 'r') as file:
            csvreader = csv.reader(file)
            header = next(csvreader)
            for row in csvreader:
                if row[3]=="TRAIN":
                    img_path = self.imgs_path + row[1]
                    class_name = row[2]+row[4]+row[5]
                    self.data.append([img_path, class_name])    
        # print(self.data)

        class_name_list =[]                     
        for img_path, class_name in self.data:
            if class_name not in class_name_list:
                class_name_list.append(class_name)

        self.class_map = {}
        tmpcnt = 0
        for class_name in class_name_list:
            self.class_map[class_name] = tmpcnt
            tmpcnt += 1 
        # print(self.class_map)
        
        ## jpg file size consistency? 

        self.img_dim   = (416, 416)  


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]        
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)        
        class_id = self.class_map[class_name]        
        img_tensor = torch.from_numpy(img) / 256
        img_tensor = img_tensor.permute(2, 0, 1)
        class_id = torch.tensor([class_id])
        return img_tensor, class_id
		

	# def __getitem__(self, idx):
	# 	
    
if __name__ == "__main__":
    dataset = CustomDataset()		
	
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    for imgs, labels in data_loader:      
        print(imgs)  
        print("Batch of images has shape: ", imgs.shape)
        print(labels)
        print("Batch of labels has shape: ", labels.shape)

	
		
		