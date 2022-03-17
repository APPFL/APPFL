import sys
import time
import logging
import argparse
import json
import torch

from appfl.config import *
from appfl.misc.data import *

import appfl.run_grpc_client as grpc_client


def main():
    # read default configuration
    cfg = OmegaConf.structured(Config)

    parser = argparse.ArgumentParser(description="Provide the configuration")
    parser.add_argument("--host", type=str, required=True)
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--use_tls", type=bool, default=False)
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--nclients", type=int, required=True)
    parser.add_argument("--logging", type=str, default="INFO")
    parser.add_argument("--api_key", default=None)
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=eval("logging." + args.logging))

    start_time = time.time()

    """ Isabelle's DenseNet (the outputs of the model are probabilities of 1 class ) """
    import importlib.machinery

    loader = importlib.machinery.SourceFileLoader("MainModel", "./IsabelleTorch.py")
    MainModel = loader.load_module()

    file = "./IsabelleTorch.pth"
    model = torch.load(file)
    model.eval()
    cfg.fed.args.loss_type = "torch.nn.BCELoss()"

    with open("../datasets/PreprocessedData/deepcovid32_train_data.json") as f:
        train_data_raw = json.load(f)

    train_dataset = Dataset(
        torch.FloatTensor(train_data_raw["x"]),
        torch.FloatTensor(train_data_raw["y"]).reshape(-1, 1),
    )

    logger = logging.getLogger(__name__)
    logger.info(
        f"----------Loaded Datasets and Model----------Elapsed Time={time.time() - start_time}"
    )

    cfg.server.host = args.host
    cfg.server.port = args.port
    cfg.server.use_tls = args.use_tls
    cfg.server.api_key = args.api_key
    logger.debug(OmegaConf.to_yaml(cfg))

    grpc_client.run_client(cfg, args.client_id, model, train_dataset)
    logger.info("------DONE------")


if __name__ == "__main__":
    main()
