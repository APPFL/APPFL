import time
import torch
import argparse
import appfl.run_mpi as rm
import appfl.run_mpi_sync as rms
from mpi4py import MPI
from appfl.config import *
from appfl.misc.data import *
from appfl.misc.utils import *
from models.flamby import flamby_train
from dataloader.flamby_dataloader import get_flamby

"""
mpiexec -np 7 python flamby_mpi_sync.py --num_epochs 5 --dataset TcgaBrca --num_local_steps 50 --server ServerFedAvg 
mpiexec -np 5 python flamby_mpi_sync.py --num_epochs 5 --dataset HeartDisease --num_local_steps 50 --server ServerFedAvg 
mpiexec -np 7 python flamby_mpi_sync.py --num_epochs 5 --dataset ISIC2019 --num_local_steps 50 --server ServerFedAvg 
mpiexec -np 4 python flamby_mpi_sync.py --num_epochs 5 --dataset IXI --num_local_steps 50 --server ServerFedAvg 
"""

## Read arguments
parser = argparse.ArgumentParser()

## device
parser.add_argument("--device", type=str, default="cpu")

## dataset
parser.add_argument("--dataset", type=str, default="TcgaBrca", choices=['TcgaBrca', 'HeartDisease', 'IXI', 'ISIC2019', 'Kits19'])

## clients
parser.add_argument("--num_clients", type=int, default=-1)
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
parser.add_argument("--server_lr", type=float, default=0.1)
parser.add_argument("--mparam_1", type=float, default=0.9)
parser.add_argument("--mparam_2", type=float, default=0.99)
parser.add_argument("--adapt_param", type=float, default=0.001)

args = parser.parse_args()

if torch.cuda.is_available():
    args.device = "cuda"

## Run
def main():
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    assert comm_size > 1, "This script requires the toal number of processes to be greater than one!"

    ## Configuration 
    cfg = OmegaConf.structured(Config)
    cfg.device = args.device
    cfg.reproduce = True
    if cfg.reproduce == True:
        set_seed(1)

    ## clients
    args.num_clients = comm_size - 1 if args.num_clients <= 0 else args.num_clients
    cfg.fed.clientname = "ClientOptim" if args.local_train_pattern == "epochs" else "ClientStepOptim"
    cfg.fed.args.num_local_steps = args.num_local_steps
    cfg.fed.args.num_local_epochs = args.num_local_epochs   

    ## server
    cfg.fed.servername = args.server
    cfg.num_epochs = args.num_epochs

    ## Specific configuration for datasets in FLamby
    train_datasets, test_dataset = get_flamby(args.dataset, args.num_clients)
    model, loss_fn, cfg.fed.args.optim, cfg.fed.args.optim_args.lr, cfg.train_data_batch_size, metric = flamby_train(args.dataset)
    cfg.test_data_batch_size = cfg.train_data_batch_size
    cfg.train_data_shuffle = True

    ## outputs
    cfg.use_tensorboard = False
    cfg.save_model_state_dict = False
    cfg.output_dirname = "./outputs_Flamby_%s_%sclients_%s_%sepochs_mpi" % (
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
    if args.num_clients == comm_size -1:
        if comm_rank == 0:
            rms.run_server(cfg, comm, model, loss_fn, args.num_clients, test_dataset, args.dataset, metric)
        else:
            rms.run_client(cfg, comm, model, loss_fn, train_datasets, test_dataset, metric)
    else:
        if comm_rank == 0:
            rm.run_server(cfg, comm, model, loss_fn, args.num_clients, test_dataset, args.dataset, metric)
        else:
            rm.run_client(cfg, comm, model, loss_fn, args.num_clients, train_datasets, test_dataset, metric)
    
    print("------DONE------", comm_rank)

if __name__ == "__main__":
    main()
