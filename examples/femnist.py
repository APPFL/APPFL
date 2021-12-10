import sys
sys.path.append("..")

import time
start_time = time.time()

## User-defined datasets
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from appfl.data import *

DataSet_name = "FEMNIST" 
num_clients = 203
num_channel = 1    # 1 if gray, 3 if color
num_classes = 62   # number of the image classes 
num_pixel   = 28   # image size = (num_pixel, num_pixel)

dir = "../datasets/RawData/%s"%(DataSet_name)

# test data for a server
test_data_raw={}  
test_data_input=[] 
test_data_label=[] 
for idx in range(36):            
    with open("%s/test/all_data_%s_niid_05_keep_0_test_9.json"%(dir, idx)) as f:    
        test_data_raw[idx] = json.load(f)    

    for client in test_data_raw[idx]["users"]:    
                                    
        for data_input in test_data_raw[idx]["user_data"][client]["x"]: 
            data_input = np.asarray(data_input)
            data_input.resize(28,28)   
            test_data_input.append([data_input])

        for data_label in test_data_raw[idx]["user_data"][client]["y"]: 
            test_data_label.append(data_label)            

test_dataset = Dataset(
                        torch.FloatTensor(test_data_input), 
                        torch.tensor(test_data_label) 
                        )

# training data for multiple clients
train_data_raw={}  
train_datasets=[]; 
for idx in range(36):            
    with open("%s/train/all_data_%s_niid_05_keep_0_train_9.json"%(dir, idx)) as f:    
        train_data_raw[idx] = json.load(f)    
    
    for client in train_data_raw[idx]["users"]:    
        
        train_data_input_resize = []
        for data_input in train_data_raw[idx]["user_data"][client]["x"]:
            data_input = np.asarray(data_input)
            data_input.resize(28,28)   
            train_data_input_resize.append([data_input])

        train_datasets.append( 
            Dataset(
            torch.FloatTensor(train_data_input_resize), 
            torch.tensor(train_data_raw[idx]["user_data"][client]["y"])
            ) 
            )
                
data_sanity_check(train_datasets, test_dataset, num_channel, num_pixel)        
                 
## User-defined model
from examples.cnn import *
model = CNN(num_channel, num_classes, num_pixel)

print("----------Loaded Datasets and Model----------Elapsed Time=",time.time()-start_time )

## Run
import appfl.run as rt
rt.main() 
 