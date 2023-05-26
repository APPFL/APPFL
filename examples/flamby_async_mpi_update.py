import time
import torch
import argparse
import appfl.run_mpi_cpas_new as rmcn
import appfl.run_mpi_async as rma
from mpi4py import MPI
from dataloader import *
from appfl.config import *
from appfl.misc.data import *
from appfl.misc.utils import *
from models.utils import flamby_train

""" read arguments """
parser = argparse.ArgumentParser()

## device
parser.add_argument("--device", type=str, default="cpu")

## dataset
parser.add_argument("--dataset", type=str, default="TcgaBrca")

## clients
parser.add_argument("--local_steps", type=int, default=50)

## server
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--server_lr", type=float, default=0.01)
parser.add_argument("--mparam_1", type=float, default=0.9)
parser.add_argument("--mparam_2", type=float, default=0.99)
parser.add_argument("--adapt_param", type=float, default=0.001)
parser.add_argument("--server", type=str, default="ServerFedAsynchronous", 
                    choices=['ServerFedAsynchronous', 
                             'ServerFedBuffer',
                             'ServerFedCPASAvgNew'
                    ])

## Fed Async
parser.add_argument("--gradient_based", action='store_true', help="Whether the algorithm requires gradient from the model")
parser.add_argument("--alpha", type=float, default=0.9, help="Mixing parameter for FedAsync Algorithm")
parser.add_argument("--staleness_func", type=str, choices=['constant', 'polynomial', 'hinge'], default='polynomial')
parser.add_argument("--a", type=float, default=0.5, help="First parameter for the staleness function")
parser.add_argument("--b", type=int, default=4, help="Second parameter for Hinge staleness function")
parser.add_argument("--K", type=int, default=3, help="Buffer size for FedBuffer algorithm")

args = parser.parse_args()

if torch.cuda.is_available():
    args.device = "cuda"

## Run
def main():
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    assert comm_size > 1, "This script requires the toal number of processes to be greater than one!"
    args.num_clients = comm_size - 1

    """ Configuration """
    cfg = OmegaConf.structured(Config(fed=FedAsync()))
    cfg.fed.clientname = "ClientOptimFlambyUpdate"

    """Specific configuration for datasets in FLamby"""
    train_datasets, test_dataset = flamby_dataset(args.dataset, args.num_clients)
    model, loss_fn, cfg.fed.args.optim, cfg.fed.args.optim_args.lr, cfg.train_data_batch_size, metric = flamby_train(args.dataset)
    cfg.test_data_batch_size = cfg.train_data_batch_size

    cfg.device = args.device
    cfg.reproduce = True
    if cfg.reproduce == True:
        set_seed(1)

    ## clients
    cfg.num_clients = args.num_clients
    cfg.fed.args.local_steps = args.local_steps
    cfg.train_data_shuffle = True

    ## server
    cfg.fed.servername = args.server
    cfg.num_epochs = args.num_epochs

    ## outputs
    cfg.use_tensorboard = False
    cfg.save_model_state_dict = False
    cfg.output_dirname = "./outputs_%s_%s_%s" % (
        args.dataset,
        args.server,
        args.num_epochs,
    )
    cfg.output_filename = "result"

    ## adaptive server
    cfg.fed.args.server_learning_rate = args.server_lr
    cfg.fed.args.server_adapt_param = args.adapt_param
    cfg.fed.args.server_momentum_param_1 = args.mparam_1
    cfg.fed.args.server_momentum_param_2 = args.mparam_2

    ## fed async/fed buffer
    cfg.fed.args.K = args.K
    cfg.fed.args.alpha = args.alpha
    cfg.fed.args.gradient_based = args.gradient_based
    cfg.fed.args.staleness_func.name = args.staleness_func
    cfg.fed.args.staleness_func.args.a = args.a
    cfg.fed.args.staleness_func.args.b = args.b

    start_time = time.time()

    print("-------Loading_Time=", time.time() - start_time)

    """ Running """
    if comm_rank == 0:
        if args.server.startswith("ServerFedCPAS"):
            rmcn.run_server(cfg, comm, model, loss_fn, args.num_clients, test_dataset, args.dataset, metric)
        else:
            rma.run_server(cfg, comm, model, loss_fn, args.num_clients, test_dataset, args.dataset, metric)
    else:
        assert comm_size == args.num_clients + 1
        if args.server.startswith("ServerFedCPAS"):
            rmcn.run_client(cfg, comm, model, loss_fn, args.num_clients, train_datasets, test_dataset, metric)
        else:
            rma.run_client(cfg, comm, model, loss_fn, args.num_clients, train_datasets, test_dataset, metric)
    
    print("------DONE------", comm_rank)

if __name__ == "__main__":
    main()

# mpiexec -np 7 python flamby_async_mpi.py --num_epochs 30
