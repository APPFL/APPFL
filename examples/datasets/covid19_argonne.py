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

    # TODO: Caclucate mean/std of Argonne dataset
    trmean = 0.5251
    trsd   = 0.1942
    temean = 0.5251
    tesd   = 0.1942
    # temean = 0.5078
    # tesd   = 0.2228

    #num_pixel = cfg.clients[client_idx].get_data.transforms.resize
    num_pixel = cfg.clients[client_idx].data_pipeline.resize
    data_dir  = cfg.clients[client_idx].data_dir

    train_transform = transforms.Compose(
        [   transforms.ToPILImage(),
            transforms.Resize(num_pixel),
            transforms.CenterCrop(num_pixel),
            transforms.ToTensor(),
            transforms.Normalize(
                # [trmean, trmean, trmean], [trsd, trsd, trsd]
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                )
        ]
    )
    test_transform = transforms.Compose(
        [   transforms.ToPILImage(),
            transforms.Resize(num_pixel),
            transforms.CenterCrop(num_pixel),
            transforms.ToTensor(),
            transforms.Normalize(
                # [temean, temean, temean], [tesd, tesd, tesd]
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                )
        ]
    )
    dataset = None
    if mode == "train":
        dataset = ArgonneCXRCovidDatset(data_dir, train_transform)
    else:
        dataset  = ArgonneCXRCovidDatset(data_dir, test_transform, mode='test')
    return dataset