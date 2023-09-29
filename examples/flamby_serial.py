import time
import torch
import argparse
import appfl.run_serial as rs
from dataloader import *
from appfl.config import *
from appfl.misc.data import *
from appfl.misc.utils import *
from models.utils import flamby_train

"""
python flamby_serial.py --num_clients 6 --num_epochs 5 --dataset TcgaBrca --num_local_steps 50 --server ServerFedAvg 
python flamby_serial.py --num_clients 4 --num_epochs 5 --dataset HeartDisease --num_local_steps 50 --server ServerFedAvg 
python flamby_serial.py --num_clients 6 --num_epochs 5 --dataset ISIC2019 --num_local_steps 50 --server ServerFedAvg 
python flamby_serial.py --num_clients 3 --num_epochs 5 --dataset IXI --num_local_steps 50 --server ServerFedAvg 
"""

## Read arguments
parser = argparse.ArgumentParser()

## device
parser.add_argument("--device", type=str, default="cpu")

## dataset
parser.add_argument("--dataset", type=str, default="TcgaBrca", choices=['TcgaBrca', 'HeartDisease', 'IXI', 'ISIC2019', 'Kits19'])

## clients
parser.add_argument("--num_clients", type=int, default=2)
parser.add_argument("--local_train_pattern", type=str, default="steps", choices=["steps", "epochs"], help="For local optimizer, what counter to use, number of steps or number of epochs")
parser.add_argument("--num_local_steps", type=int, default=100)
parser.add_argument("--num_local_epochs", type=int, default=1)

## server
parser.add_argument("--server", type=str, default="ServerFedAvg", 
                    choices=[
                        "ServerFedAvg", 
                        "ServerFedAvgMomentum",
                        "ServerFedAdam",
                        "ServerFedAdagrad",
                        "ServerFedYogi"
                    ])
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--server_lr", type=float, default=0.01)
parser.add_argument("--mparam_1", type=float, default=0.9)
parser.add_argument("--mparam_2", type=float, default=0.99)
parser.add_argument("--adapt_param", type=float, default=0.001)

args = parser.parse_args()

if torch.cuda.is_available():
    args.device = "cuda"

## Run
def main():
    ## Configuration 
    cfg = OmegaConf.structured(Config)
    cfg.device = args.device
    cfg.reproduce = True
    if cfg.reproduce == True:
        set_seed(1)

    ## clients
    cfg.num_clients = args.num_clients
    cfg.fed.clientname = "ClientOptim" if args.local_train_pattern == "epochs" else "ClientStepOptim"
    cfg.fed.args.num_local_steps = args.num_local_steps
    cfg.fed.args.num_local_epochs = args.num_local_epochs   

    ## server
    cfg.fed.servername = args.server
    cfg.num_epochs = args.num_epochs

    ## Specific configuration for datasets in FLamby
    train_datasets, test_dataset = flamby_dataset(args.dataset, args.num_clients)
    model, loss_fn, cfg.fed.args.optim, cfg.fed.args.optim_args.lr, cfg.train_data_batch_size, metric = flamby_train(args.dataset)
    cfg.test_data_batch_size = cfg.train_data_batch_size
    cfg.train_data_shuffle = True

    ## outputs
    cfg.use_tensorboard = False
    cfg.save_model_state_dict = False
    cfg.output_dirname = "./outputs_Flamby_%s_%sclients_%s_%sepochs_serial" % (
        args.dataset,
        args.num_clients,
        args.server,
        args.num_epochs,
    )
    cfg.output_filename = "result"

    ## adaptive server
    cfg.fed.args.server_learning_rate = args.server_lr
    cfg.fed.args.server_adapt_param = args.adapt_param
    cfg.fed.args.server_momentum_param_1 = args.mparam_1
    cfg.fed.args.server_momentum_param_2 = args.mparam_2

    start_time = time.time()

    print("-------Loading_Time=", time.time() - start_time)

    ## Running 
    rs.run_serial(cfg, model, loss_fn, train_datasets, test_dataset, args.dataset, metric)
    
    print("------DONE------")

if __name__ == "__main__":
    main()
