import time
import torch
import argparse
import appfl.run_mpi as rm
from mpi4py import MPI
from dataloader import *
from appfl.config import *
from appfl.misc.data import *
from appfl.misc.utils import *
from models.utils import get_model 
 
""" read arguments """ 

parser = argparse.ArgumentParser() 

## device
parser.add_argument('--device', type=str, default="cpu")    

## dataset and model
parser.add_argument('--dataset', type=str, default="CIFAR10")   
parser.add_argument('--num_channel', type=int, default=3)   
parser.add_argument('--num_classes', type=int, default=10)   
parser.add_argument('--num_pixel', type=int, default=32)   
parser.add_argument('--pretrained', type=int, default=-1)   
parser.add_argument('--model', type=str, default="resnet18_new")   
parser.add_argument('--train_data_batch_size', type=int, default=128)   
parser.add_argument('--test_data_batch_size', type=int, default=128)   
parser.add_argument("--partition", type=str, default="iid", 
                    choices=["iid", "partition_noiid", "dirichlet_noiid"])
parser.add_argument("--seed", type=int, default=42)

## clients
parser.add_argument('--client_optimizer', type=str, default="SGD")    
parser.add_argument('--client_lr', type=float, default=0.1)    
parser.add_argument('--client_lr_mom', type=float, default=0.9)
parser.add_argument('--local_steps', type=int, default=200)    

## server
parser.add_argument("--server", type=str, default="ServerFedAvg", 
                    choices=["ServerFedAvg", "ServerFedAvgMomentum"])
parser.add_argument('--num_epochs', type=int, default=20)    
parser.add_argument("--server_lr", type=float, default=0.01)
parser.add_argument("--mparam_1", type=float, default=0.9)
parser.add_argument("--mparam_2", type=float, default=0.99)
parser.add_argument("--adapt_param", type=float, default=0.001)


## Simulation
parser.add_argument("--do_simulation", action="store_true", help="Whether to do client local training-time simulation")
parser.add_argument("--simulation_distrib", type=str, default="normal", choices=["normal", "exp", "homo"], help="Local trianing-time distribution for different clients")
parser.add_argument("--avg_tpb", type=float, default=0.5, help="Average time-per-batch for clint local trianing-time simulation")
parser.add_argument("--global_std_scale", type=float, default=0.3, help="Std scale for time-per-batch for different clients")
parser.add_argument("--exp_scale", type=float, default=0.5, help="Scale for exponential distribution")
parser.add_argument("--exp_bin_size", type=float, default=0.2, help="Width of the bin when discretizing the client tbp in exponential distribution")
parser.add_argument("--local_std_scale", type=float, default=0.05, help="Std scale for time-per-batch for different experiments of one client")
parser.add_argument("--delta_warmup", action="store_true", help="When running the code on delta, we need to first warm up the computing resource")

args = parser.parse_args()

if torch.cuda.is_available():
    args.device="cuda"

## Run
def main():
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size() 

    assert comm_size > 1, "This script requires the toal number of processes to be greater than one!"
    args.num_clients = comm_size - 1

    ## Reproducibility
    set_seed(1)

    """ Configuration """     
    cfg = OmegaConf.structured(Config) 

    cfg.device = args.device
    cfg.save_model_state_dict = False

    ## dataset
    cfg.train_data_batch_size = args.train_data_batch_size
    cfg.test_data_batch_size = args.test_data_batch_size
    cfg.train_data_shuffle = True

    ## clients
    cfg.num_clients = args.num_clients
    cfg.fed.args.optim = args.client_optimizer
    cfg.fed.args.optim_args.lr = args.client_lr
    cfg.fed.args.optim_args.momentum = args.client_lr_mom
    cfg.fed.args.local_steps = args.local_steps
    cfg.train_data_shuffle = True
    cfg.fed.clientname = "ClientOptimUpdate"
    
    ## server
    cfg.fed.servername = args.server
    cfg.num_epochs = args.num_epochs

    ## outputs        
    cfg.use_tensorboard = False
    cfg.save_model_state_dict = False
    cfg.output_dirname = "./outputs_%s_%s_%sClients_%s_%s_%sEpochs" % (
        args.dataset,
        args.partition,
        args.num_clients,
        args.simulation_distrib if args.do_simulation else "noSim",
        args.server,
        args.num_epochs,
    )
    cfg.output_filename = "result"

    ## adaptive server
    cfg.fed.args.server_learning_rate = args.server_lr
    cfg.fed.args.server_adapt_param = args.adapt_param
    cfg.fed.args.server_momentum_param_1 = args.mparam_1
    cfg.fed.args.server_momentum_param_2 = args.mparam_2

    ## simulation
    cfg.fed.args.do_simulation = args.do_simulation
    cfg.fed.args.simulation_distrib = args.simulation_distrib
    cfg.fed.args.avg_tpb = args.avg_tpb
    cfg.fed.args.global_std_scale = args.global_std_scale
    cfg.fed.args.local_std_scale = args.local_std_scale
    cfg.fed.args.exp_scale = args.exp_scale
    cfg.fed.args.exp_bin_size = args.exp_bin_size
    cfg.fed.args.seed = args.seed
    cfg.fed.args.delta_warmup = args.delta_warmup

    start_time = time.time()

    """ User-defined model """
    model = get_model(args)
    loss_fn = torch.nn.CrossEntropyLoss()   

    """ User-defined data """
    train_datasets, test_dataset = eval(args.partition)(comm, cfg, args.dataset, seed=args.seed, alpha1=args.num_clients)

    ## Sanity check for the user-defined data
    if cfg.data_sanity == True:
        data_sanity_check(train_datasets, test_dataset, args.num_channel, args.num_pixel)

    print("-------Loading_Time=", time.time() - start_time)

    """ Running """
    if comm_rank == 0:
        rm.run_server(cfg, comm, model, loss_fn, args.num_clients, test_dataset, args.dataset)
    else:
        assert comm_size == args.num_clients + 1
        rm.run_client(cfg, comm, model, loss_fn, args.num_clients, train_datasets, test_dataset)
    
    print("------DONE------", comm_rank)

if __name__ == "__main__":
    main()

# mpiexec -np 7 python cifar_sync_mpi_update.py --num_epochs 12 --do_simulation
