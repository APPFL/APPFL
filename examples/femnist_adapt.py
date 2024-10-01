import time
import torch
import argparse
from appfl.config import *
from appfl.misc.data import *
from appfl.misc.utils import *
from models.cnn import *
import appfl.run_serial_adapt as rs
import appfl.run_mpi_adapt as rm
from mpi4py import MPI
from models.utils import get_model
from dataloader.femnist_dataloader import get_femnist

""" read arguments """

parser = argparse.ArgumentParser()

parser.add_argument("--device", type=str, default="cpu")

## dataset
parser.add_argument("--dataset", type=str, default="FEMNIST")
parser.add_argument("--num_channel", type=int, default=1)
parser.add_argument("--num_classes", type=int, default=62)
parser.add_argument("--num_pixel", type=int, default=28)
parser.add_argument("--model", type=str, default="resnet18-legacy")

## algorithm
parser.add_argument("--federation_type", type=str, default="Federated")  ## Federated, ICEADMM, IIADMM

## clients
parser.add_argument("--client_name", type=str, default="ClientAdaptOptim")  ## ClientOptim  
parser.add_argument("--num_clients", type=int, default=1)
parser.add_argument("--client_optimizer", type=str, default="Adam")
parser.add_argument("--client_lr", type=float, default=0.001)
parser.add_argument("--num_local_epochs", type=int, default=3)

## server
# parser.add_argument("--server", type=str, default="ServerFedAvg")  
# changed to AdaptiveFLServer
parser.add_argument("--server", type=str, default="AdaptiveFLServer")  
parser.add_argument("--num_epochs", type=int, default=10)

parser.add_argument("--server_lr", type=float, required=False)
parser.add_argument("--mparam_1", type=float, required=False)
parser.add_argument("--mparam_2", type=float, required=False)
parser.add_argument("--adapt_param", type=float, required=False)

parser.add_argument("--pretrained", type=int, default=0)

## privacy preserving
parser.add_argument("--use_dp", action="store_true", default=False, help="Whether to enable differential privacy technique to preserve privacy")
parser.add_argument("--epsilon", type=float, default=1, help="Privacy budget - stronger privacy as epsilon decreases")
parser.add_argument("--clip_grad", action="store_true", default=False, help="Whether to clip the gradients")
parser.add_argument("--clip_value", type=float, default=1.0, help="Max norm of the gradients")
parser.add_argument("--clip_norm", type=float, default=1, help="Type of the used p-norm for gradient clipping")

args = parser.parse_args()

if torch.cuda.is_available():
    args.device = "cuda"

def main():
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # read default configuration
    cfg = OmegaConf.structured(Config)

    ## Reproducibility
    cfg.reproduce = True
    if cfg.reproduce == True:
        set_seed(1)

    start_time = time.time()
    train_datasets, test_dataset = get_femnist(args.num_pixel, args.num_channel, args.pretrained)

    if cfg.data_sanity == True:
        data_sanity_check(
            train_datasets, test_dataset, args.num_channel, args.num_pixel
        )

    args.num_clients = len(train_datasets)
    
    
    
    model = get_model(args)
    loss_fn = torch.nn.CrossEntropyLoss()
    print(
        "----------Loaded Datasets and Model----------Elapsed Time=",
        time.time() - start_time,
    )

    ## settings
    cfg.device = args.device
    cfg.num_clients = args.num_clients
    cfg.num_epochs = args.num_epochs
 
    cfg.fed = eval(args.federation_type+"()")    
    cfg.fed.clientname = args.client_name
    cfg.fed.servername = args.server     # updates

    

    if args.federation_type == "Federated":
        cfg.fed.args.optim = args.client_optimizer
        cfg.fed.args.optim_args.lr = args.client_lr
        cfg.fed.servername = args.server
        cfg.fed.args.num_local_epochs = args.num_local_epochs
      
    ## privacy preserving
    cfg.fed.args.use_dp = args.use_dp
    cfg.fed.args.epsilon = args.epsilon
    cfg.fed.args.clip_grad = args.clip_grad
    cfg.fed.args.clip_value = args.clip_value
    cfg.fed.args.clip_norm = args.clip_norm

    ## outputs
    cfg.use_tensorboard = True

    if comm_size > 1:
        if comm_rank == 0:
            rm.run_server(
                cfg, comm, model, loss_fn, args.num_clients, test_dataset, args.dataset
            )
        else:            
            rm.run_client(cfg, comm, model, loss_fn, args.num_clients, train_datasets)
        print("------DONE------", comm_rank)
    else:
        rs.run_serial(cfg, model, loss_fn, train_datasets, test_dataset, args.dataset)


if __name__ == "__main__":
    main()


# To run CUDA-aware MPI:
# mpiexec -np 5 --mca opal_cuda_support 1 python ./femnist.py
# To run MPI:
# mpiexec -np 5 python ./femnist.py
# To run:
# python ./femnist.py
# To run with resnet pretrained weight:
# python ./femnist.py --model resnet18 --pretrained 1

# Experiment settings
# mpiexec -np 5 python ./femnist_mpi.py --client_optimizer=Adam --client_lr=1e-3 --num_local_epochs=5 --num_epochs=10 --server=ServerFedAvg 
# mpiexec -np 5 python ./femnist_mpi.py --client_optimizer=Adam --client_lr=1e-3 --num_local_epochs=5 --num_epochs=10 --server=AdaptiveFLServer


# mpiexec -np 5 python ./femnist_mpi.py --client_optimizer=Adam --client_lr=1e-3 --num_local_epochs=5 --num_epochs=10 --server=ServerFedAvg --use_dp --epsilon=1.0
# mpiexec -np 5 python ./femnist_mpi.py --client_optimizer=Adam --client_lr=1e-3 --num_local_epochs=5 --num_epochs=10 --server=ServerAdaptiveFL --use_dp --epsilon=1.0


# mpiexec -np 5 python ./femnist_adapt.py --client_name=ClientAdaptOptim --client_lr=1e-3 --num_local_epochs=5 --num_epochs=10 --server=AdaptiveFLServer