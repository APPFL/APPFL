def get_data(
    cfg,
    client_idx: int,
    mode = "train"
):
    import cv2
    import os
    import os.path as osp
    import csv
    from appfl.misc.data import Dataset
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    import torch
    import numpy as np
    from glob import glob

    class ArgonneCXRCovidDatset(Dataset):
        def __init__(self, data_dir, transform, mode="train"):
            assert mode in ["train", "test", "val"]
            self.datadir = data_dir
            self.img_dir = osp.join(self.datadir, "data")
            self.annot_file = osp.join(self.datadir, "%s_%s.txt" % (cfg.clients[client_idx].data_pipeline.split_file, mode))
            self.data_list  = [] 
            self.labels     = []
            skip=1
            with open(self.annot_file, "r") as fi:
                rd = csv.reader(fi, delimiter=' ')
                for i, row in enumerate(rd):
                    if i % skip == 0:
                        self.data_list.append(row[1])
                        assert row[2] in ['negative', 'positive']
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
        
    pixel_resize = cfg.clients[client_idx].data_pipeline.resize
    pixel_crop = cfg.clients[client_idx].data_pipeline.center_crop
    data_dir  = cfg.clients[client_idx].data_dir

    train_transform =[   
            transforms.ToPILImage(),
            transforms.Resize(pixel_resize),
            transforms.CenterCrop(pixel_crop),
            transforms.ToTensor(),]
    if "train_mean" in cfg.clients[client_idx].data_pipeline:
        train_transform.append(transforms.Normalize(
                cfg.clients[client_idx].data_pipeline.train_mean,
                cfg.clients[client_idx].data_pipeline.train_std
            ))
    train_transform = transforms.Compose(train_transform)

    val_transform =[   
        transforms.ToPILImage(),
            transforms.Resize(pixel_resize),
            transforms.CenterCrop(pixel_crop),
            transforms.ToTensor(),]
    if "train_mean" in cfg.clients[client_idx].data_pipeline:
        val_transform.append(transforms.Normalize(
                cfg.clients[client_idx].data_pipeline.train_mean,
                cfg.clients[client_idx].data_pipeline.train_std
            ))
    val_transform = transforms.Compose(val_transform)

    test_transform = [   
            transforms.ToPILImage(),
            transforms.Resize(pixel_resize),
            transforms.CenterCrop(pixel_crop),
            transforms.ToTensor(),
        ]
    if "test_mean" in cfg.clients[client_idx].data_pipeline:
        test_transform.append(transforms.Normalize(
                cfg.clients[client_idx].data_pipeline.test_mean,
                cfg.clients[client_idx].data_pipeline.test_std
            ))
    test_transform = transforms.Compose(test_transform)
    
    dataset = None
    if mode == "train":
        dataset = ArgonneCXRCovidDatset(data_dir, train_transform)
    elif mode == "val":
        dataset  = ArgonneCXRCovidDatset(data_dir, val_transform, mode='val')
    else:
        dataset  = ArgonneCXRCovidDatset(data_dir, test_transform, mode='test')
    return dataset