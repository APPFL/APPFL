import json
import numpy as np
import torch

 
class FEMNISTTrain():
    def __init__(self, in_features, num_classes, pixel): 
        dir = "../../../datasets/FEMNIST/train"

        train_data_raw={}  
        train_data_image={}  ## { clients : tensor(image) }
        train_data_class={}  ## { clients : tensor(class) }
        self.dataloader = []
        for idx in range(36):            
            with open(dir+"/all_data_%s_niid_05_keep_0_train_9.json"%(idx)) as f:    
                train_data_raw[idx] = json.load(f)    
            
            for client in train_data_raw[idx]["users"]:    
                train_data_image[client] = []
                
                for image_data in train_data_raw[idx]["user_data"][client]["x"]:                    
                    image_data = np.asarray(image_data)
                    image_data.resize(pixel,pixel)                    
                    train_data_image[client].append([image_data])
                                
                train_data_image[client] = torch.FloatTensor(train_data_image[client])
                train_data_class[client] = torch.tensor(train_data_raw[idx]["user_data"][client]["y"])

                self.dataloader.append( (train_data_image[client], train_data_class[client]) )
         
        self.num_clients = len(train_data_image.keys())
 
              
 
class FEMNISTTest():
    def __init__(self, in_features, num_classes, pixel): 
        dir = "../../../datasets/FEMNIST/test"

        test_data_raw={}  
        test_data_image={}  ## { clients : tensor(image) }
        test_data_class={}  ## { clients : tensor(class) }
        self.dataloader = []
        for idx in range(36):            
            with open(dir+"/all_data_%s_niid_05_keep_0_test_9.json"%(idx)) as f:    
                test_data_raw[idx] = json.load(f)    
            
            for client in test_data_raw[idx]["users"]:    
                test_data_image[client] = []
                
                for image_data in test_data_raw[idx]["user_data"][client]["x"]:                    
                    image_data = np.asarray(image_data)
                    image_data.resize(pixel,pixel)                    
                    test_data_image[client].append([image_data])
                                
                test_data_image[client] = torch.FloatTensor(test_data_image[client])
                test_data_class[client] = torch.tensor(test_data_raw[idx]["user_data"][client]["y"])

                self.dataloader.append( (test_data_image[client], test_data_class[client]) )
         
        self.num_clients = len(test_data_image.keys())
 