import os
import time
from turtle import forward

import numpy as np
import torch

import torchvision
from torchvision.transforms import ToTensor

from appfl.config import *
from appfl.misc.data import *
from appfl.misc.utils import *
from models.cnn import *

import appfl.run_serial as rs
from torchvision import transforms
import argparse

""" read arguments """

parser = argparse.ArgumentParser()

parser.add_argument("--device", type=str, default="cpu")

## dataset
parser.add_argument("--dataset", type=str, default="MNIST")
parser.add_argument("--num_channel", type=int, default=1)
parser.add_argument("--num_classes", type=int, default=10)
parser.add_argument("--num_pixel", type=int, default=28)

## clients
parser.add_argument("--num_clients", type=int, default=1)
parser.add_argument("--client_optimizer", type=str, default="SGD")
parser.add_argument("--client_lr", type=float, default=1e-3)
parser.add_argument("--num_local_epochs", type=int, default=1)

## server
parser.add_argument("--server", type=str, default="ServerFedAvg")
parser.add_argument("--num_epochs", type=int, default=3)

parser.add_argument("--server_lr", type=float, required=False)
parser.add_argument("--mparam_1", type=float, required=False)
parser.add_argument("--mparam_2", type=float, required=False)
parser.add_argument("--adapt_param", type=float, required=False)


args = parser.parse_args()

if torch.cuda.is_available():
    args.device = "cuda"

#### MODEL
class DenseNet121(nn.Module):
        """
        DenseNet121 model with additional Sigmoid layer for classification
        """
        def __init__(self, num_output):
            super(DenseNet121, self).__init__()
            self.densenet121 = torchvision.models.densenet121(pretrained = True)
            num_features = self.densenet121.classifier.in_features
            self.densenet121.classifier = nn.Sequential(
                nn.Linear(num_features, num_output)
            )
        def forward(self, x):
            x = self.densenet121(x)
            return x

#### Dataset
import csv
import os.path as osp
import cv2

class ArgonneCXRCovidDatset(Dataset):
    def __init__(self, data_dir, transform, mode='train'):
        assert mode in ['train', 'test']
        self.datadir = data_dir
        self.img_dir = osp.join(self.datadir, mode)
        self.annot_file = osp.join(self.datadir, "%s.txt" % mode)
        self.data_list  = [] 
        self.labels     = []
        skip=10
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

num_pixel=224
data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
            transforms.Resize(num_pixel),
            transforms.CenterCrop(num_pixel),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
       transforms.ToPILImage(),
            transforms.Resize(num_pixel),
            transforms.CenterCrop(num_pixel),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def get_data():
    data_dir = '/mnt/data0-nfs/shared-datasets/anl-covid-xray/archive/'

    # Root download the data if not already available.
    # test data for a server
    # test_data_raw = eval("torchvision.datasets." + args.dataset)(
    #     dir, download=True, train=False, transform=ToTensor()
    # )
    test_data_raw = ArgonneCXRCovidDatset(data_dir, data_transforms['test'], 'test')

    test_data_input = []
    test_data_label = []
    for idx in range(len(test_data_raw)):
        test_data_input.append(test_data_raw[idx][0].tolist())
        test_data_label.append(test_data_raw[idx][1])

    test_dataset = Dataset(
        torch.FloatTensor(test_data_input), torch.tensor(test_data_label)
    )

    train_data_raw = ArgonneCXRCovidDatset(data_dir, data_transforms['train'], 'train')

    train_datasets = [train_data_raw]
    
    # split_train_data_raw = np.array_split(range(len(train_data_raw)), args.num_clients)
    
    # for i in range(args.num_clients):

    #     train_data_input = []
    #     train_data_label = []
    #     for idx in split_train_data_raw[i]:
    #         train_data_input.append(train_data_raw[idx][0].tolist())
    #         train_data_label.append(train_data_raw[idx][1])

    #     train_datasets.append(
    #         Dataset(
    #             torch.FloatTensor(train_data_input),
    #             torch.tensor(train_data_label),
    #         )
    #     )
    return train_datasets, test_dataset


def get_model():
    ## User-defined model
    model = DenseNet121(2)
    return model

def appl_fix_loss(output, target):
    return torch.nn.BCEWithLogitsLoss()(output, target.unsqueeze(-1))

## Run
def main():

    """ Configuration """
    cfg = OmegaConf.structured(Config)

    cfg.device = args.device
    cfg.reproduce = True
    if cfg.reproduce == True:
        set_seed(1)

    ## clients
    cfg.num_clients = args.num_clients
    cfg.fed.args.optim = args.client_optimizer
    cfg.fed.args.optim_args.lr = args.client_lr
    cfg.fed.args.num_local_epochs = args.num_local_epochs

    ## server
    cfg.fed.servername = args.server
    cfg.num_epochs = args.num_epochs

    ## outputs

    cfg.use_tensorboard = False

    cfg.save_model_state_dict = False

    cfg.output_dirname = "./outputs_%s_%s_%s" % (
        args.dataset,
        args.server,
        args.client_optimizer,
    )
    if args.server_lr != None:
        cfg.fed.args.server_learning_rate = args.server_lr
        cfg.output_dirname += "_ServerLR_%s" % (args.server_lr)

    if args.adapt_param != None:
        cfg.fed.args.server_adapt_param = args.adapt_param
        cfg.output_dirname += "_AdaptParam_%s" % (args.adapt_param)

    if args.mparam_1 != None:
        cfg.fed.args.server_momentum_param_1 = args.mparam_1
        cfg.output_dirname += "_MParam1_%s" % (args.mparam_1)

    if args.mparam_2 != None:
        cfg.fed.args.server_momentum_param_2 = args.mparam_2
        cfg.output_dirname += "_MParam2_%s" % (args.mparam_2)

    cfg.output_filename = "result"

    start_time = time.time()

    """ User-defined model """
    model = get_model()
    # loss_fn = appl_fix_loss
    loss_fn = torch.nn.CrossEntropyLoss()   

    ## loading models
    cfg.load_model = False
    if cfg.load_model == True:
        cfg.load_model_dirname = "./save_models"
        cfg.load_model_filename = "Model"
        model = load_model(cfg)

    """ User-defined data """
    train_datasets, test_dataset = get_data()

    ## Sanity check for the user-defined data
    if cfg.data_sanity == True:
        data_sanity_check(
            train_datasets, test_dataset, args.num_channel, args.num_pixel
        )

    print(
        "-------Loading_Time=",
        time.time() - start_time,
    )

    """ saving models """
    cfg.save_model = False
    if cfg.save_model == True:
        cfg.save_model_dirname = "./save_models"
        cfg.save_model_filename = "Model"

    """ Running """
    rs.run_serial(cfg, model, loss_fn, train_datasets, test_dataset, args.dataset)
        


if __name__ == "__main__":
    main()

# To run:
# python ./mnist_no_mpi.py
