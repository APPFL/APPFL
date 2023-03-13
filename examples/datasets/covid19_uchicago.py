def get_data(
    cfg,
    client_idx: int,
    mode = "train"
):
    import cv2
    import os
    import pandas as pd
    from appfl.misc.data import Dataset
    from torch.utils.data import DataLoader
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
            try:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except:
                raise RuntimeError("File broken " + img_path)
            image = self.transform(image)
            
            if self.truth_df.examCOVIDstatus.iloc[idx] == 'Negative':
                label = 0 #torch.FloatTensor([0])
            else:
                label = 1 #torch.FloatTensor([1])
            return image, label

        def run_stats(self, batch_size=1024, num_workers=4):
            N = len(self)
            dtld = DataLoader(self, batch_size=batch_size,shuffle=False, num_workers=num_workers, drop_last = False)
            _mean = torch.zeros(3)
            _std = torch.zeros(3)
            _max = torch.Tensor([-1e8, -1e8, -1e8])
            _min = torch.Tensor([1e8, 1e8, 1e8])
            n_hist_bins = 255
            hist_range = (-3.0, 3.0)
            _hist = torch.zeros(n_hist_bins)

            for img, lbl in dtld:
                _mean += img.view(len(img),3,-1).mean(dim=-1).sum(0)/N
                _std += img.view(len(img),3,-1).std(dim=-1).sum(0)/N
                _max = torch.maximum(img.view(len(img),3,-1).amax(-1).amax(0), _max)
                _min = torch.minimum(img.view(len(img),3,-1).amin(-1).amin(0), _min)
                _hist, _bins = torch.histogram(img[:,0,...], range = hist_range, density=True, bins=n_hist_bins) #
                _hist += _hist * img.shape[0]/N

            return {"mean": _mean.tolist(), "std": _std.tolist(),
                    "max": _max.tolist(), "min": _min.tolist(),
                    "hist": _hist.tolist(), "bins": _bins.tolist()
                    }

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
        
    train_transform = [   
            transforms.ToPILImage(),
            transforms.Resize(num_pixel),
            transforms.CenterCrop(num_pixel),    
            transforms.ToTensor(),
        ]
    if "train_mean" in cfg.clients[client_idx].data_pipeline:
        train_transform.append(transforms.Normalize(
                cfg.clients[client_idx].data_pipeline.train_mean,
                cfg.clients[client_idx].data_pipeline.train_std
            ))
    train_transform = transforms.Compose(train_transform)
    
    test_transform = [   
            transforms.ToPILImage(),
            transforms.Resize(num_pixel),
            transforms.CenterCrop(num_pixel),
            transforms.ToTensor(),
        ]
    if "test_mean" in cfg.clients[client_idx].data_pipeline:
        test_transform.append(transforms.Normalize(
                cfg.clients[client_idx].data_pipeline.test_mean,
                cfg.clients[client_idx].data_pipeline.test_std
            ))
    test_transform = transforms.Compose(test_transform)

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