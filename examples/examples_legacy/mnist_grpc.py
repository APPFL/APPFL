"""
To run grpc with 5 clients:
mpiexec -np 6 python ./mnist_grpc.py \
    --partition class_noiid \
    --loss_fn losses/celoss.py \
    --loss_fn_name CELoss \
    --num_epochs 10

To run grpc with 5 clients and Naive authenticator:
mpiexec -np 6 python ./mnist_grpc.py \
    --use_ssl \
    --use_authenticator \
    --authenticator Naive \
    --partition class_noiid \
    --loss_fn losses/celoss.py \
    --loss_fn_name CELoss \
    --num_epochs 10
"""

import time
import torch
import argparse
from mpi4py import MPI
from omegaconf import OmegaConf
from appfl.config import Config
from appfl.misc.utils import set_seed
from appfl.misc.data import data_sanity_check
from losses.utils import get_loss
from models.utils import get_model
from metric.utils import get_metric
import appfl.run_grpc_server as grpc_server
import appfl.run_grpc_client as grpc_client
from dataloader.mnist_dataloader import get_mnist
from appfl.comm.grpc import (
    load_credential_from_file,
    ROOT_CERTIFICATE,
    SERVER_CERTIFICATE,
    SERVER_CERTIFICATE_KEY,
)


## read arguments
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cpu")

## dataset
parser.add_argument("--dataset", type=str, default="MNIST")
parser.add_argument("--num_channel", type=int, default=1)
parser.add_argument("--num_classes", type=int, default=10)
parser.add_argument("--num_pixel", type=int, default=28)
parser.add_argument("--model", type=str, default="CNN")
parser.add_argument(
    "--partition",
    type=str,
    default="iid",
    choices=["iid", "class_noiid", "dirichlet_noiid"],
)
parser.add_argument("--seed", type=int, default=42)

## clients
parser.add_argument("--client_optimizer", type=str, default="Adam")
parser.add_argument("--client_lr", type=float, default=1e-3)
parser.add_argument(
    "--local_train_pattern",
    type=str,
    default="steps",
    choices=["steps", "epochs"],
    help="For local optimizer, what counter to use, number of steps or number of epochs",
)
parser.add_argument("--num_local_steps", type=int, default=100)
parser.add_argument("--num_local_epochs", type=int, default=1)

## server
parser.add_argument("--server", type=str, default="ServerFedAvg")
parser.add_argument("--num_epochs", type=int, default=2)
parser.add_argument("--server_lr", type=float, default=0.1)
parser.add_argument("--mparam_1", type=float, default=0.9)
parser.add_argument("--mparam_2", type=float, default=0.99)
parser.add_argument("--adapt_param", type=float, default=0.001)

## privacy preserving
parser.add_argument(
    "--use_dp",
    action="store_true",
    default=False,
    help="Whether to enable differential privacy technique to preserve privacy",
)
parser.add_argument(
    "--epsilon",
    type=float,
    default=1,
    help="Privacy budget - stronger privacy as epsilon decreases",
)
parser.add_argument(
    "--clip_grad",
    action="store_true",
    default=False,
    help="Whether to clip the gradients",
)
parser.add_argument(
    "--clip_value", type=float, default=1.0, help="Max norm of the gradients"
)
parser.add_argument(
    "--clip_norm",
    type=float,
    default=1,
    help="Type of the used p-norm for gradient clipping",
)

## loss function
parser.add_argument(
    "--loss_fn",
    type=str,
    required=False,
    help="path to the custom loss function definition file, use cross-entropy loss by default if no path is specified",
)
parser.add_argument(
    "--loss_fn_name",
    type=str,
    required=False,
    help="class name for the custom loss in the loss function definition file, choose the first class by default if no name is specified",
)

## evaluation metric
parser.add_argument(
    "--metric",
    type=str,
    default="metric/acc.py",
    help="path to the custom evaluation metric function definition file, use accuracy by default if no path is specified",
)
parser.add_argument(
    "--metric_name",
    type=str,
    required=False,
    help="function name for the custom eval metric function in the metric function definition file, choose the first function by default if no name is specified",
)

## grpc communication
parser.add_argument("--use_ssl", action="store_true", default=False)
parser.add_argument("--use_authenticator", action="store_true", default=False)
parser.add_argument("--uri", type=str, default="localhost:50051")
parser.add_argument("--server_certificate_key", type=str, default="default")
parser.add_argument("--server_certificate", type=str, default="default")
parser.add_argument("--root_certificates", type=str, default="default")
parser.add_argument(
    "--authenticator", type=str, default="Globus", choices=["Globus", "Naive"]
)
parser.add_argument(
    "--globus_group_id", type=str, default="77c1c74b-a33b-11ed-8951-7b5a369c0a53"
)

args = parser.parse_args()

if torch.cuda.is_available():
    args.device = "cuda"


## Run
def main():
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    assert (
        comm_size > 1
    ), "This script requires the toal number of processes to be greater than one!"
    args.num_clients = comm_size - 1

    ## Configuration
    cfg = OmegaConf.structured(Config)
    cfg.device = args.device
    cfg.reproduce = True
    if cfg.reproduce:
        set_seed(1)

    ## GRPC configurations
    cfg.uri = args.uri
    cfg.use_ssl = args.use_ssl
    cfg.use_authenticator = args.use_authenticator
    cfg.server.server_certificate_key = (
        load_credential_from_file(args.server_certificate_key)
        if args.server_certificate_key != "default"
        else SERVER_CERTIFICATE_KEY
    )
    cfg.server.server_certificate = (
        load_credential_from_file(args.server_certificate)
        if args.server_certificate != "default"
        else SERVER_CERTIFICATE
    )
    cfg.client.root_certificates = (
        load_credential_from_file(args.root_certificates)
        if args.root_certificates != "default"
        else ROOT_CERTIFICATE
    )
    if args.authenticator == "Globus":
        cfg.server.authenticator_kwargs.is_fl_server = True
        cfg.server.authenticator_kwargs.globus_group_id = args.globus_group_id
        cfg.client.authenticator_kwargs.is_fl_server = False
    else:
        cfg.server.authenticator_kwargs = {}
        cfg.client.authenticator_kwargs = {}
    cfg.authenticator = args.authenticator + "Authenticator"

    ## clients
    cfg.num_clients = comm_size - 1
    cfg.fed.clientname = (
        "ClientOptim" if args.local_train_pattern == "epochs" else "ClientStepOptim"
    )
    cfg.fed.args.optim = args.client_optimizer
    cfg.fed.args.optim_args.lr = args.client_lr
    cfg.fed.args.num_local_steps = args.num_local_steps
    cfg.fed.args.num_local_epochs = args.num_local_epochs

    ## server
    cfg.fed.servername = args.server
    cfg.num_epochs = args.num_epochs

    ## outputs
    cfg.output_dirname = f"./outputs_{args.dataset}_{args.partition}_{args.num_clients}clients_{args.server}_{args.num_epochs}epochs_grpc"

    ## adaptive server
    cfg.fed.args.server_learning_rate = args.server_lr  # FedAdam
    cfg.fed.args.server_adapt_param = args.adapt_param  # FedAdam
    cfg.fed.args.server_momentum_param_1 = args.mparam_1  # FedAdam, FedAvgm
    cfg.fed.args.server_momentum_param_2 = args.mparam_2  # FedAdam

    ## privacy preserving
    cfg.fed.args.use_dp = args.use_dp
    cfg.fed.args.epsilon = args.epsilon
    cfg.fed.args.clip_grad = args.clip_grad
    cfg.fed.args.clip_value = args.clip_value
    cfg.fed.args.clip_norm = args.clip_norm

    start_time = time.time()

    ## User-defined model
    model = get_model(args)
    loss_fn = get_loss(args.loss_fn, args.loss_fn_name)
    metric = get_metric(args.metric, args.metric_name)

    ## User-defined data
    train_datasets, test_dataset = get_mnist(
        comm,
        num_clients=cfg.num_clients,
        partition=args.partition,
        visualization=True,
        output_dirname=cfg.output_dirname,
        seed=args.seed,
        alpha1=args.num_clients,
    )

    ## Sanity check for the user-defined data
    if cfg.data_sanity:
        data_sanity_check(
            train_datasets, test_dataset, args.num_channel, args.num_pixel
        )

    print("-------Loading_Time=", time.time() - start_time)

    if comm_size > 1:
        # Try to launch both a server and clients.
        if comm_rank == 0:
            grpc_server.run_server(
                cfg, model, loss_fn, cfg.num_clients, test_dataset, metric
            )
        else:
            grpc_client.run_client(
                cfg,
                comm_rank - 1,
                model,
                loss_fn,
                train_datasets[comm_rank - 1],
                comm_rank,
                test_dataset,
                metric,
            )

        print("------DONE------", comm_rank)
    else:
        # Just launch a server.
        grpc_server.run_server(cfg, model, cfg.num_clients, test_dataset)


if __name__ == "__main__":
    main()
