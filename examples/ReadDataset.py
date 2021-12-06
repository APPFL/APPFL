import json
import torch
from torch.utils.data import Dataset, DataLoader
 
class Test(Dataset):
    def __init__(self,dir):                             
        
        with open("%s/all_test_data.json"%(dir)) as f:    
            test_data_raw = json.load(f)    

        self.test_data_image = torch.FloatTensor(test_data_raw["x"])
        self.test_data_class = torch.tensor(test_data_raw["y"])     

    def __len__(self):
        return len(self.test_data_class)
    
    def __getitem__(self, idx):
        return self.test_data_image[idx], self.test_data_class[idx]

class Train(Dataset):
    def __init__(self,dir,client):                             
        
        with open("%s/all_train_data_client_%s.json"%(dir, client)) as f:    
            test_data_raw = json.load(f)    

        self.test_data_image = torch.FloatTensor(test_data_raw["x"])
        self.test_data_class = torch.tensor(test_data_raw["y"])     

    def __len__(self):
        return len(self.test_data_class)
    
    def __getitem__(self, idx):
        return self.test_data_image[idx], self.test_data_class[idx]

def ReadDataset(DataSet_name, num_clients, num_channel, num_pixel):
    
    dir = "../datasets/ProcessedData/%s_Clients_%s"%(DataSet_name,num_clients)

    ## Datasets 
    test_dataset = Test(dir) 
    train_datasets=[]; 
    for client in range(num_clients):    
        train_datasets.append(Train(dir,client))

    ## Check if "DataLoader" from PyTorch works.
    train_dataloader = DataLoader(train_datasets[0], batch_size=64, shuffle=False)    
    for image, class_id in train_dataloader:
        # print("image=", image.shape, " class_id=", class_id.shape)
        assert(image.shape[0] == class_id.shape[0])
        assert(image.shape[1] == num_channel)
        assert(image.shape[2] == num_pixel)
        assert(image.shape[3] == num_pixel)

    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)    
    for image, class_id in test_dataloader:
        # print("image=", image.shape, " class_id=", class_id.shape)
        assert(image.shape[0] == class_id.shape[0])
        assert(image.shape[1] == num_channel)
        assert(image.shape[2] == num_pixel)
        assert(image.shape[3] == num_pixel)     

    return train_datasets, test_dataset



