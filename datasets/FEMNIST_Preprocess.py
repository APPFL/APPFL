import numpy as np
import os
import json
import torch
 
## Read Test Data 
test_data_raw={}  
test_data_image=[] 
test_data_class=[] 
for idx in range(36):            
    with open("./RawData/FEMNIST/test/all_data_%s_niid_05_keep_0_test_9.json"%(idx)) as f:    
        test_data_raw[idx] = json.load(f)    
    for client in test_data_raw[idx]["users"]:                                
        for image_data in test_data_raw[idx]["user_data"][client]["x"]:                    
            image_data = np.asarray(image_data)
            image_data.resize(28,28)   
            test_data_image.append([image_data])

        for class_data in test_data_raw[idx]["user_data"][client]["y"]:                          
            test_data_class.append(class_data)

test_data_image = torch.FloatTensor(test_data_image)

## Read Train Data
train_data_raw={}  
train_data_image={}  
train_data_class={}  
for idx in range(36):            
    with open("./RawData/FEMNIST/train/all_data_%s_niid_05_keep_0_train_9.json"%(idx)) as f:    
        train_data_raw[idx] = json.load(f)    
    for client in train_data_raw[idx]["users"]:    
        train_data_image[client] = []        
        
        for image_data in train_data_raw[idx]["user_data"][client]["x"]:                    
            image_data = np.asarray(image_data)
            image_data.resize(28,28)   
            train_data_image[client].append([image_data])
                        
        train_data_image[client] = torch.FloatTensor(train_data_image[client])
        train_data_class[client] = train_data_raw[idx]["user_data"][client]["y"]

num_clients = len(train_data_class) 

## Output Directories
dir = "./ProcessedData/FEMNIST_Clients_{}".format(num_clients) 
if os.path.isdir(dir) == False:
    os.mkdir(dir)    

## Testing Data (Server)
out_test_file  = dir+"/all_test_data.json"
all_test_data = {}
all_test_data["x"] = test_data_image.tolist()
all_test_data["y"] = test_data_class    

with open(out_test_file, 'w') as outfile:
    json.dump(all_test_data, outfile)

## Training Data (Clients)
tmpcnt = 0
for client in train_data_image.keys():    

    print("--client ", tmpcnt)

    user_data = {};      
    user_data["x"] = train_data_image[client].tolist()         
    user_data["y"] = train_data_class[client]   

    out_train_file = dir+"/all_train_data_client_{}.json".format(tmpcnt)

    with open(out_train_file, 'w') as outfile:
        json.dump(user_data, outfile)
    
    tmpcnt += 1