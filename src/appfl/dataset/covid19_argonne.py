def get_data(
    cfg,
    client_idx: int
):
    import cv2
    import os
    import os.path as osp
    import csv
    from appfl.misc.data import Dataset
    import torchvision.transforms as transforms
    import torch
    import numpy as np
    from glob import glob
    
    class ArgonneCXRCovidDatset(Dataset):
        def __init__(self, data_dir, transform, mode='train'):
            assert mode in ['train', 'test']
            self.datadir = data_dir
            self.img_dir = osp.join(self.datadir, mode)
            self.annot_file = osp.join(self.datadir, "%s.txt" % mode)
            self.data_list  = [] 
            self.labels     = []
            with open(self.annot_file, "r") as fi:
                rd = csv.reader(fi, delimiter=' ')
                for row in rd:
                    self.data_list.append(row[1])
                    self.labels.append(0 if row[2] == 'negative' else 1)
            self.transform = transform
            
        def __len__(self):
            return len(self.data_list)
        
        def __getitem__(self, idx):
            img_path = os.path.join(self.img_dir, self.data_list[idx])
            image = cv2.imread(img_path) #NEEDS TO BE (3,32,32)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.transform(image)
            label = self.labels[idx]
            return image, label
    
    # TODO: Caclucate mean/std of Argonne dataset
    trmean = 0.6181
    trsd = 0.2510
    temean = 0.6250
    tesd = 0.2498
    
    num_pixel = cfg.clients[client_idx].get_data.transforms.resize
    data_dir  = cfg.clients[client_idx].data_dir

    train_transform = transforms.Compose(
        [   transforms.ToPILImage(),
            transforms.Resize(num_pixel),
            transforms.CenterCrop(num_pixel),
            transforms.ToTensor(),
            transforms.Normalize([trmean, trmean, trmean], [trsd, trsd, trsd])
        ]
    )
    test_transform = transforms.Compose(
        [   transforms.ToPILImage(),
            transforms.Resize(num_pixel),
            transforms.CenterCrop(num_pixel),
            transforms.ToTensor(),
            transforms.Normalize([temean, temean, temean], [tesd, tesd, tesd])
        ]
    )
    train_dataset = ArgonneCXRCovidDatset(data_dir, train_transform)
    test_dataset  = ArgonneCXRCovidDatset(data_dir, test_transform, mode='test')
    return train_dataset

# if __name__ == '__main__':
#     from omegaconf import OmegaConf
#     train_data, test_data = get_data(OmegaConf.create({
#         "data_dir": "/eagle/covid-xray/archive/",
#         "num_pixel" : 224
#         }), 0)
#     print(len(train_data), len(test_data))