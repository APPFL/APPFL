import sys
import time
import logging

## User-defined datasets
import json
import torch

from appfl.config import *
from appfl.misc.data import *

import appfl.run_grpc_server as grpc_server
import appfl.run_grpc_client as grpc_client
from mpi4py import MPI

DataSet_name = "DeepCovid"
num_clients = 2


def main():
    # read default configuration
    cfg = OmegaConf.structured(Config)
    cfg.num_epochs = 10

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    ## Reproducibility
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()

    """ Isabelle's DenseNet (the outputs of the model are probabilities of 1 class ) """
    import importlib.machinery

    loader = importlib.machinery.SourceFileLoader(
        "MainModel", "./gcloud/IsabelleTorch.py"
    )
    MainModel = loader.load_module()

    file = "./gcloud/IsabelleTorch.pth"
    model = torch.load(file)
    model.eval()
    cfg.fed.args.loss_type = "torch.nn.BCELoss()"

    with open("./datasets/PreprocessedData/deepcovid32_train_data.json") as f:
        train_data_raw_1 = json.load(f)
    with open("./datasets/PreprocessedData/deepcovid32_test_data.json") as f:
        train_data_raw_2 = json.load(f)

    train_datasets = []
    train_datasets.append(
        Dataset(
            torch.FloatTensor(train_data_raw_1["x"]),
            torch.FloatTensor(train_data_raw_1["y"]).reshape(-1, 1),
        )
    )
    train_datasets.append(
        Dataset(
            torch.FloatTensor(train_data_raw_2["x"]),
            torch.FloatTensor(train_data_raw_2["y"]).reshape(-1, 1),
        )
    )

    # test data
    with open("./datasets/PreprocessedData/deepcovid32_test_data.json") as f:
        test_data_raw = json.load(f)

    test_dataset = Dataset(
        torch.FloatTensor(test_data_raw["x"]),
        torch.FloatTensor(test_data_raw["y"]).reshape(-1, 1),
    )

    print(
        "----------Loaded Datasets and Model----------Elapsed Time=",
        time.time() - start_time,
    )
    """ saving models """
    cfg.save_model = True
    if cfg.save_model == True:
        cfg.save_model_dirname = "./save_models"
        cfg.save_model_filename = "Covid_Binary_Isabelle_FedAvg"
        cfg.checkpoints_interval = 2

    if comm_size > 1:
        # Try to launch both a server and clients.
        if comm_rank == 0:
            grpc_server.run_server(cfg, model, num_clients, test_dataset)
        else:
            grpc_client.run_client(cfg, comm_rank, model, train_datasets[comm_rank - 1])
        print("------DONE------", comm_rank)
    else:
        # Just launch a server.
        grpc_server.run_server(cfg, model, num_clients, test_dataset)


if __name__ == "__main__":
    main()
