def get_data(cfg, client_idx: int):
    #supports chexpert
    from appfl.misc.data import Dataset
    import csv
    import pandas as pd
    import os

    class FromCSVDataset(Dataset):

        def __init__(self, path, start_line = -1, end_line = -1):
            '''
            path: path of CSV file which records location of x-rays and their labelings
            start_line: Starting line of the csv file from where data would be part of this object
            end_line:   endling line of the csv file till where data would be part of this object   

            start_line and end_line is used for segregating data for each client
            '''
            xrays = []
            labels = []
            
            with open(path) as paths_file:
                csv_ = csv.reader(paths_file)

                #skipping header line
                next(csv_, None)

                #iterating until start_line
                if start_line != -1:
                    for _ in range(start_line):
                        next(csv_, None)

                num_iter_left = end_line - start_line
                for line in csv_:
                    if num_iter_left <= 0 and start_line != -1: break
                    num_iter_left -= 1
                    xray_path = line[0] 
                    label = line[5:]
                    #assuming uncertain or not reported data as negative
                    for idx, label_ in enumerate(label):
                        if label_: label[idx] = float(label_) 
                        if not label_ or  float(label_) == -1: label[idx] = 0
                    xrays.append(os.getcwd() + "/Data/" + xray_path)
                    labels.append(label)
            self.xrays = xrays
            self.labels = labels
            
        
        def __getitem__(self, idx):
            
            transforms_list = []
            transforms_list.append(transforms.Resize((224, 224)))  #DenseNet takes 224*224 images
            transforms_list.append(transforms.ToTensor())
            transform = transforms.Compose(transforms_list)

            xray_path = self.xrays[idx]
            xray = transform(Image.open(xray_path).convert('RGB'))       #Although grayscale, DenseNet takes 3 channel
            label = torch.FloatTensor(self.labels[idx])
            
            return xray, label

        def __len__(self):
            return len(self.xrays)
    
    data_dir = cfg.clients[client_idx].data_dir #TODO: data is already in data_dir? 

    train_data = pd.read_csv(os.path.join(data_dir, "train.csv")) 
    len_training_data = len(train_data)
    split_size = int(len_training_data/cfg.num_clients)

    train_dataset = FromCSVDataset(data_dir, split_size * client_idx, split_size * (client_idx + 1))

    return train_dataset


def get_model():
    import torchvision
    import torch.nn as nn
    ## User-defined model
    class DenseNet121(nn.Module):
        """
        DenseNet121 model with additional Sigmoid layer for classification
        TODO: See improvements in model and scope of using lateral X-rays by merging two densenet models (ensemble training).
        """
        def __init__(self, num_output = 14):
            super(DenseNet121, self).__init__()
            self.densenet121 = torchvision.models.densenet121(pretrained = False)
            num_features = self.densenet121.classifier.in_features
            self.densenet121.classifier = nn.Sequential(
                nn.Linear(num_features, num_output),
                nn.Sigmoid()
            )

        def forward(self, x):
            x = self.densenet121(x)
            return x
    return DenseNet121