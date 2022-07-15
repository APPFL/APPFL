def get_data(
    cfg,
    client_idx: int
):
    import cv2
    import os
    import pandas as pd
    from appfl.misc.data import Dataset
    import torchvision.transforms as transforms
    import torch
    import numpy as np

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
                label = torch.FloatTensor([0])
            else:
                label = torch.FloatTensor([1])
            return image, label

    if cfg.num_pixel==32:
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
            transforms.Resize(cfg.num_pixel),
            transforms.CenterCrop(cfg.num_pixel),
            #transforms.RandomResizedCrop(args.num_pixel),            
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            #norm values when num_pixel=32
            transforms.Normalize([trmean, trmean, trmean], [trsd, trsd, trsd])
        ]
    )
    test_transform = transforms.Compose(
        [   transforms.ToPILImage(),
            transforms.Resize(cfg.num_pixel),
            transforms.CenterCrop(cfg.num_pixel),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            #norm values when num_pixel=32
            transforms.Normalize([temean, temean, temean], [tesd, tesd, tesd])
        ]
    )
  

    ## Dataset
    # train_dataset = UChicagoCXRCovidDatset(main_path = './ImageData',
    #                                        annotations_file = './train_annotations',
    #                                        transform = train_transform)
    data_dir = cfg.clients[client_idx].data_dir
    train_dataset = UChicagoCXRCovidDatset(main_path = data_dir,
                                           annotations_file = './train_annotations',
                                           transform = train_transform)
    
    split_train_data_raw = np.array_split(range(len(train_dataset)), cfg.num_clients)        
    # train_datasets = []
    # for i in range(cfg.num_clients):
    #     train_datasets.append(torch.utils.data.Subset(train_dataset, split_train_data_raw[i]))

    train_dataset = torch.utils.data.Subset(train_dataset, split_train_data_raw[client_idx])
    
    # TODO: do we need test_dataset in client?  
    # test_dataset = UChicagoCXRCovidDatset(main_path = './ImageData',
    #                                        annotations_file = './test_annotations',
    #                                        transform = test_transform)
    test_dataset = UChicagoCXRCovidDatset(main_path = data_dir,
                                           annotations_file = './test_annotations',
                                           transform = test_transform)
    
    return train_dataset, test_dataset