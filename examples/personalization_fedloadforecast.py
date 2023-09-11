import os
import time
import argparse
import numpy as np
import torch
from mpi4py import MPI
import copy

from appfl.config import *
from appfl.misc.data import *
from appfl.misc.utils import load_model_state
import appfl.run_serial as rs
import appfl.run_mpi as rm

from models.cnn import *
from models.utils import get_model
from models.utils import validate_parameter_names

from losses.utils import get_loss
from metric.utils import get_metric

from datasets.PreprocessedData.NREL_Preprocess import NRELDataDownloader

import warnings
warnings.filterwarnings("ignore",category=UserWarning)

""" define functions for custom data type in argparses"""

def list_of_strings(arg):
    return arg.split(',')

def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x

def unique(list1):
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

""" read arguments """

parser = argparse.ArgumentParser()

## device
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

## dataset
parser.add_argument("--dataset", type=str, default="NRELCA")
parser.add_argument("--n_features", type=int, default=8)
parser.add_argument("--n_lookback", type=int, default=12)
parser.add_argument("--n_lstm_layers", type=int, default=2)
parser.add_argument("--n_hidden_size", type=int, default=20)
parser.add_argument("--model", type=str, default="LSTM")
parser.add_argument("--train_test_boundary", type=restricted_float, default=0.8)

## clients
parser.add_argument("--num_clients", type=int, default=1)
parser.add_argument("--client_optimizer", type=str, default="Adam")
parser.add_argument("--client_lr", type=float, default=1e-3)
parser.add_argument("--num_local_epochs", type=int, default=4)

## server
parser.add_argument("--server", type=str, default="ServerFedAvg")
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--server_lr", type=float, required=False)
parser.add_argument("--mparam_1", type=float, required=False)
parser.add_argument("--mparam_2", type=float, required=False)
parser.add_argument("--adapt_param", type=float, required=False)

## model load and save
parser.add_argument("--save_model", type=int, default=1)
parser.add_argument("--save_every", type=int, default=4)
parser.add_argument("--load_model", type=int, default=0)
parser.add_argument("--load_model_suffix", type=str, default="Round_10")

## loss function and evaluation metric
parser.add_argument("--loss_fn", type=str, default='losses/mseloss.py')
parser.add_argument("--metric", type=str, default='metric/mae.py')

## personalization
parser.add_argument("--personalization_layers", type=list_of_strings, default=[])
parser.add_argument("--personalization_config_name", type=str, default = "")

# parse args
args = parser.parse_args()

# directory where to save dataset
dir = os.getcwd() + "/datasets/PreprocessedData"

def get_data(comm: MPI.Comm):

    ## sanity checks
    if args.dataset not in ['NRELCA','NRELIL','NRELNY']:
        raise ValueError('Currently only NREL data for CA, IL, NY are supported. Please modify download script if you want a different state.')         
    
    state_idx = args.dataset[4:]
    if not os.path.isfile(dir+'/%sdataset.npz'%args.dataset):
        # building ID's whose data to download
        if state_idx == 'CA':
            b_id = [15227, 15233, 15241, 15222, 15225, 15228, 15404, 15429, 15460, 15445, 15895, 16281, 16013, 16126, 16145, 47395, 15329, 15504, 15256, 15292, 15294, 15240, 15302, 15352, 15224, 15231, 15243, 17175, 17215, 18596, 15322, 15403, 15457, 15284, 15301, 15319, 15221, 15226, 15229, 15234, 15237, 15239]
        if state_idx == 'IL':
            b_id =  [108872, 109647, 110111, 108818, 108819, 108821, 108836, 108850, 108924, 108879, 108930, 108948, 116259, 108928, 109130, 113752, 115702, 118613, 108816, 108840, 108865, 108888, 108913, 108942, 108825, 108832, 108837, 109548, 114596, 115517, 109174, 109502, 109827, 108846, 108881, 108919, 108820, 108823, 108828, 108822, 108864, 108871]
        if state_idx == 'NY':
            b_id = [205362, 205863, 205982, 204847, 204853, 204854, 204865, 204870, 204878, 205068, 205104, 205124, 205436, 213733, 213978, 210920, 204915, 205045, 204944, 205129, 205177, 204910, 205024, 205091, 204849, 204860, 204861, 208090, 210116, 211569, 204928, 204945, 205271, 204863, 204873, 204884, 204842, 204843, 204844, 204867, 204875, 204880]
        downloader = NRELDataDownloader(dset_tags = [state_idx], b_id = [b_id])
        downloader.download_data()
        downloader.save_data(fname=dir+'/NREL%sdataset.npz'%state_idx)
        
    dset_raw = np.load(dir+'/NREL%sdataset.npz'%state_idx)['data']
    if args.num_clients > dset_raw.shape[0]:
        raise ValueError('More clients requested than present in dataset.')
    if args.n_features != dset_raw.shape[-1]:
        raise ValueError('Incorrect number of features passed as argument, the number of features present in dataset is %d.'%dset_raw.shape[-1])
    
    ## process dataset
    dset_reduced_clients = dset_raw[:np.minimum(dset_raw.shape[0],args.num_clients),:,:].copy()
    dset_train = dset_reduced_clients[:,:int(args.train_test_boundary*dset_reduced_clients.shape[1]),:].copy()
    dset_test = dset_reduced_clients[:,int(args.train_test_boundary*dset_reduced_clients.shape[1]):,:].copy()
    # scale to [0,1]
    for idx_f in range(dset_train.shape[-1]):
        feature_minval = dset_train[:,:,idx_f].min()
        feature_maxval = dset_train[:,:,idx_f].max()
        dset_train[:,:,idx_f] = (dset_train[:,:,idx_f] - feature_minval) / (feature_maxval - feature_minval)
        dset_test[:,:,idx_f] = (dset_test[:,:,idx_f] - feature_minval) / (feature_maxval - feature_minval)
    # record as train and test datasets
    # TODO: ensure that there is enough temporal data which is greater than lookback
    train_datasets = []
    test_inputs = []
    test_labels = []
    for idx_cust in range(dset_train.shape[0]):
        # train dataset entries
        train_inputs = []
        train_labels = []
        for idx_time in range(dset_train.shape[1]-args.n_lookback):
            train_inputs.append(dset_train[idx_cust,idx_time:idx_time+args.n_lookback,:])
            train_labels.append(dset_train[idx_cust,idx_time+args.n_lookback,0])
        dset = Dataset(torch.FloatTensor(np.array(train_inputs)),torch.FloatTensor(np.array(train_labels)))
        train_datasets.append(dset)
        # test dataset entries
        for idx_time in range(dset_test.shape[1]-args.n_lookback):
            test_inputs.append(dset_test[idx_cust,idx_time:idx_time+args.n_lookback,:])
            test_labels.append(dset_train[idx_cust,idx_time+args.n_lookback,0])
    test_dataset = Dataset(torch.FloatTensor(np.array(test_inputs)),torch.FloatTensor(np.array(test_labels)))

    return train_datasets, test_dataset
    

def main():

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # read default configuration
    cfg = OmegaConf.structured(Config)

    ## Reproducibility
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()
    train_datasets, test_dataset = get_data(comm)
    
    ## Model
    model = get_model(args)    
    loss_fn = get_loss(args.loss_fn, None)
    metric = get_metric(args.metric, None)
    
    ## If personalization is used, validate personalization
    args.personalization_layers = unique(args.personalization_layers)
    is_valid,is_empty = validate_parameter_names(model,args.personalization_layers)
    if not is_valid:
        raise TypeError('The arguments containing names of personalization layers are invalid for the current model.')
    else:
        if not is_empty:
            cfg.personalization = True
            cfg.p_layers = args.personalization_layers
            cfg.config_name = args.personalization_config_name
        else:
            cfg.personalization = False
            cfg.p_layers = []

    # print(
    #     "----------Loaded Datasets and Model----------Elapsed Time=",
    #     time.time() - start_time,
    # ) 

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
    
    ## save/load
    cfg.output_dirname = "./outputs_%s_%s_%s_%s" % (
        args.dataset,
        args.server,
        args.client_optimizer,
        args.personalization_config_name
    )
    if args.save_model:
        cfg.save_model_state_dict = True
        cfg.save_model = True
        cfg.save_model_dirname = "./save_models_NREL_%s"%args.personalization_config_name
        cfg.save_model_filename = "model_%s_%s_%s"%(args.dataset,args.client_optimizer,args.server)
    if args.load_model:
        if cfg.personalization:
            model_clients = [copy.deepcopy(model) for _ in range(args.num_clients)]
            cfg.load_model_dirname = "./save_models_NREL_%s"%args.personalization_config_name
            cfg.load_model_filename = "model_%s_%s_%s_%s"%(args.dataset,args.client_optimizer,args.server,args.load_model_suffix)
            load_model_state(cfg,model)
            for c_idx in range(args.num_clients):
                load_model_state(cfg,model_clients[c_idx],client_id=c_idx)
        else:
            cfg.save_model_dirname = "./save_models_NREL_%s"%args.personalization_config_name
            cfg.save_model_filename = "model_%s_%s_%s"%(args.dataset,args.client_optimizer,args.server) 
            model = load_model(cfg)

    if comm_size > 1:
        if comm_rank == 0:
            rm.run_server(cfg, comm, model, loss_fn, args.num_clients, test_dataset, args.dataset, metric)
        else:
            if cfg.personalization and args.load_model:
                model_to_send = model_clients
            else:
                model_to_send = model
            rm.run_client(cfg, comm, model_to_send, loss_fn, args.num_clients, train_datasets, test_dataset, metric)
        print("------DONE------", comm_rank)
    else:
        rs.run_serial(cfg, model, loss_fn, train_datasets, test_dataset, args.dataset)


if __name__ == "__main__": 
    main()

# To run CUDA-aware MPI:
# mpiexec -np 2 --mca opal_cuda_support 1 python ./celeba.py
# To run MPI:
# mpiexec -np 2 python ./celeba.py
# To run:
# python ./celeba.py
# To run with resnet pretrained weight:
# python ./celeba.py --model resnet18 --pretrained 1