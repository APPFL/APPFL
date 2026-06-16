"""Load AFL-Lib's generated .npz shards into APPFL (S9c controlled comparison).

Using AFL-Lib's exact data files guarantees IDENTICAL data, partition (dirichlet alpha),
and input normalization (AFL saves tensors already normalized to [-1,1]) across both libs.

AFL-Lib layout: <afl_dir>/dataset/<dataset>/{train,test}/<client_id>.npz
each npz: np.load(...)['data'] -> {'x': [...], 'y': [...]}
"""

import os
import numpy as np
import torch
from appfl.misc.data import Dataset


def _load_npz(path):
    with open(path, "rb") as f:
        d = np.load(f, allow_pickle=True)["data"].tolist()
    x = torch.tensor(np.array(d["x"]), dtype=torch.float32)
    y = torch.tensor(np.array(d["y"]), dtype=torch.int64)
    return Dataset(x, y)


def get_afl_data(num_clients, client_id, afl_dir, dataset="mnist", **kwargs):
    """Per-client (train_shard, test_shard) — exactly AFL-Lib's shards for this client."""
    base = os.path.join(afl_dir, "dataset", dataset)
    train = _load_npz(os.path.join(base, "train", f"{client_id}.npz"))
    test = _load_npz(os.path.join(base, "test", f"{client_id}.npz"))
    return train, test


def get_afl_test_all(afl_dir, dataset="mnist", num_clients=10, **kwargs):
    """Concatenation of all clients' test shards = AFL-Lib's full test set
    (what AFL's test_all evaluates the global model on). For server-side global eval."""
    base = os.path.join(afl_dir, "dataset", dataset, "test")
    xs, ys = [], []
    for cid in range(num_clients):
        with open(os.path.join(base, f"{cid}.npz"), "rb") as f:
            d = np.load(f, allow_pickle=True)["data"].tolist()
        xs.append(np.array(d["x"]))
        ys.append(np.array(d["y"]))
    x = torch.tensor(np.concatenate(xs), dtype=torch.float32)
    y = torch.tensor(np.concatenate(ys), dtype=torch.int64)
    return Dataset(x, y)
