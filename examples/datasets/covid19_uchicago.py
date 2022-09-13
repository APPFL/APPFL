def get_data(
    cfg,
    client_idx: int,
    mode = "train"
):
    import cv2
    import os
    import pandas as pd
    from appfl.misc.data import Dataset
    import torchvision.transforms as transforms
    import torch
    import numpy as np
    import os.path as osp

    class UChicagoCXRCovidDatset(Dataset):
        def __init__(self, main_path, annotations_file, transform):
            self.main_path = main_path
            self.truth_df = pd.read_csv(annotations_file)
            self.transform = transform
            
        def __len__(self):
            return len(self.truth_df)
        
        def __getitem__(self, idx):
            img_path = os.path.join(self.main_path, self.truth_df.fmtImName_sd.iloc[idx])
            image = cv2.imread(img_path) #NEEDS TO BE (3,32,32)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.transform(image)
            
            if self.truth_df.examCOVIDstatus.iloc[idx] == 'Negative':
                label = 0 #torch.FloatTensor([0])
            else:
                label = 1 #torch.FloatTensor([1])
            return image, label

    num_pixel = cfg.clients[client_idx].data_pipeline.num_pixels

    if num_pixel==32:
        trmean = 0.6181
        trsd = 0.2398
        temean = 0.6250
        tesd = 0.2384
    else:
        trmean = 0.6181
        trsd = 0.2510
        temean = 0.6250
        tesd = 0.2498
        
    train_transform = transforms.Compose(
        [   transforms.ToPILImage(),
            transforms.Resize(num_pixel),
            transforms.CenterCrop(num_pixel),
            #transforms.RandomResizedCrop(args.num_pixel),            
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            #norm values when num_pixel=32
            # transforms.Normalize([trmean, trmean, trmean], [trsd, trsd, trsd])
        ]
    )
    test_transform = transforms.Compose(
        [   transforms.ToPILImage(),
            transforms.Resize(num_pixel),
            transforms.CenterCrop(num_pixel),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            #norm values when num_pixel=32
            # transforms.Normalize([temean, temean, temean], [tesd, tesd, tesd])
        ]
    )
    ## Dataset
    data_dir = cfg.clients[client_idx].data_dir
    dataset = None
    
    if mode == "train":
        #train dataset
        dataset = UChicagoCXRCovidDatset(main_path = data_dir,
                                            annotations_file = cfg.clients[client_idx].data_pipeline.train_annotation_dir,
                                            transform = train_transform)
    elif mode == "val":
        dataset = UChicagoCXRCovidDatset(main_path = data_dir,
                                           annotations_file = cfg.clients[client_idx].data_pipeline.val_annotation_dir,
                                           transform = test_transform) 
    else:
        dataset = UChicagoCXRCovidDatset(main_path = data_dir,
                                           annotations_file = cfg.clients[client_idx].data_pipeline.test_annotation_dir,
                                           transform = test_transform)    
    return dataset