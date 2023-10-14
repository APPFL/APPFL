import os
import json
import torch
from appfl.misc.data import Dataset, data_sanity_check

def get_corona(args):
    dir = os.getcwd() + f"/datasets/PreprocessedData/Coronahack_Clients_{args.num_clients}"

    # test data for a server
    with open("%s/all_test_data.json" % (dir)) as f:
        test_data_raw = json.load(f)
    test_dataset = Dataset(torch.FloatTensor(test_data_raw["x"]), torch.tensor(test_data_raw["y"]))

    # training data for multiple clients
    train_datasets = []
    for client in range(args.num_clients):
        with open("%s/all_train_data_client_%s.json" % (dir, client)) as f:
            train_data_raw = json.load(f)
        train_datasets.append(
            Dataset(
                torch.FloatTensor(train_data_raw["x"]),
                torch.tensor(train_data_raw["y"]),
            )
        )

    data_sanity_check(train_datasets, test_dataset, args.num_channel, args.num_pixel)
    return train_datasets, test_dataset

