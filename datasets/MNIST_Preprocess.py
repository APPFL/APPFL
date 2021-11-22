import numpy as np
import torchvision
from torchvision.transforms import ToTensor
import os
import json

train_data_raw = torchvision.datasets.MNIST(
    f"./RawData",
    download=True,
    train=True,
    transform=ToTensor()
)

test_data_raw = torchvision.datasets.MNIST(
    f"./RawData",
    download=True,
    train=False,
    transform=ToTensor()
)

num_clients = 4  ## Any Integer

## Output Directories
dir = "./ProcessedData/MNIST_Clients_{}".format(num_clients) 
if os.path.isdir(dir) == False:
    os.mkdir(dir)    

## Testing Data (Server)
out_test_file  = dir+"/all_test_data.json"
all_test_data = {}
all_test_data["x"] = []
all_test_data["y"] = []
for idx in range(len(test_data_raw)):    
    all_test_data["x"].append( test_data_raw[idx][0].tolist() )
    all_test_data["y"].append( test_data_raw[idx][1] )
with open(out_test_file, 'w') as outfile:
    json.dump(all_test_data, outfile)

## Training Data (Clients)
split_train_data_raw = np.array_split(range(len(train_data_raw)), num_clients)     
for i in range(num_clients):
    print("--client ", i)

    user_data = {}; x=[]; y=[]    
    for idx in split_train_data_raw[i]:        
        x.append(train_data_raw[idx][0].tolist())
        y.append(train_data_raw[idx][1])
    user_data["x"] = x        
    user_data["y"] = y    

    out_train_file = dir+"/all_train_data_client_{}.json".format(i)

    with open(out_train_file, 'w') as outfile:
        json.dump(user_data, outfile)
 