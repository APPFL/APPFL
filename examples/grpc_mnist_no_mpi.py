import argparse
import logging
import pathlib
import numpy as np
import sys
import time
import torch
import torchvision

from models.cnn import CNN
from torchvision.transforms import ToTensor

# Local imports
sys.path.append(f"{pathlib.Path(__file__).parent.absolute()}/../src")
import appfl.run_grpc_server as grpc_server
import appfl.run_grpc_client as grpc_client

from appfl.config import Config, OmegaConf
from appfl.misc.data import data_sanity_check, Dataset


def get_data(args):
    if args.run_server:
        logging.info("Running as server")
        # test data for a server
        test_data_raw = eval("torchvision.datasets." + args.dataset)(
            f"./datasets/RawData", download=True, train=False, transform=ToTensor()
        )
    else:
        logging.info(f"Running as client {args.client_id}")
        # test data for a client
        test_data_raw = eval("torchvision.datasets." + args.dataset)(
            f"./datasets/RawData", download=False, train=False, transform=ToTensor()
        )

    test_data_input = []
    test_data_label = []
    for idx in range(len(test_data_raw)):
        test_data_input.append(test_data_raw[idx][0].tolist())
        test_data_label.append(test_data_raw[idx][1])

    test_dataset = Dataset(
        torch.FloatTensor(test_data_input), torch.tensor(test_data_label)
    )

    # training data for multiple clients
    train_data_raw = eval("torchvision.datasets." + args.dataset)(
        f"./datasets/RawData", download=False, train=True, transform=ToTensor()
    )

    split_train_data_raw = np.array_split(range(len(train_data_raw)), args.num_clients)
    train_datasets = []
    for i in range(args.num_clients):

        train_data_input = []
        train_data_label = []
        for idx in split_train_data_raw[i]:
            train_data_input.append(train_data_raw[idx][0].tolist())
            train_data_label.append(train_data_raw[idx][1])

        train_datasets.append(
            Dataset(
                torch.FloatTensor(train_data_input),
                torch.tensor(train_data_label),
            )
        )

    return train_datasets, test_dataset


def get_model(args):
    ## User-defined model
    model = CNN(args.num_channel, args.num_classes, args.num_pixel)
    return model


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")

    ## dataset
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--num_channel", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--num_pixel", type=int, default=28)

    ## client options
    parser.add_argument("-c", dest="client_id", type=int, default=1)
    parser.add_argument("--client_optimizer", type=str, default="Adam")
    parser.add_argument("--client_lr", type=float, default=1e-3)
    parser.add_argument("--num_local_epochs", type=int, default=3)

    ## clients
    parser.add_argument("--num_clients", type=int, default=1)

    ## server
    parser.add_argument("-s", dest="run_server", action="store_true", default=False)
    parser.add_argument("--server", type=str, default="ServerFedAvg")
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--server_lr", type=float, required=False)
    parser.add_argument("--mparam_1", type=float, required=False)
    parser.add_argument("--mparam_2", type=float, required=False)
    parser.add_argument("--adapt_param", type=float, required=False)

    args = parser.parse_args()

    """ Configuration """
    cfg = OmegaConf.structured(Config)
    cfg.device = args.device

    ## clients
    cfg.num_clients = args.num_clients
    cfg.fed.args.optim = args.client_optimizer
    cfg.fed.args.optim_args.lr = args.client_lr
    cfg.fed.args.num_local_epochs = args.num_local_epochs

    ## server
    cfg.fed.servername = args.server
    cfg.num_epochs = args.num_epochs

    ## outputs

    cfg.output_dirname = (
        f"./outputs_{args.dataset}_{args.server}_{args.client_optimizer}"
    )

    cfg.output_filename = "result"

    start_time = time.time()

    """ User-defined model """
    model = get_model(args)
    loss_fn = torch.nn.CrossEntropyLoss()

    cfg.validation = False
    cfg.load_model = False

    """ User-defined data """
    train_datasets, test_dataset = get_data(args)

    ## Sanity check for the user-defined data
    if cfg.data_sanity:
        data_sanity_check(
            train_datasets, test_dataset, args.num_channel, args.num_pixel
        )

    logging.info(
        f"-------Loading_Time={time.time() - start_time} "
        f"({args.client_id if not args.run_server else 'Server'})"
    )

    """ saving models """
    cfg.save_model = False

    if args.run_server:
        logging.info("------server------")
        grpc_server.run_server(cfg, model, loss_fn, args.num_clients)
    else:
        logging.info("------client------")
        grpc_client.run_client(
            cfg,
            args.client_id - 1,
            model,
            loss_fn,
            train_datasets[args.client_id - 1],
            args.client_id - 1,
            test_dataset,
        )

        logging.info("------DONE------")


if __name__ == "__main__":
    main()
