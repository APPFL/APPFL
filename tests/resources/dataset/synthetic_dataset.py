import torch
from appfl.misc.data import Dataset


def get_synthetic(
    num_clients: int,
    client_id: int,
    num_samples: int = 48,
    num_classes: int = 10,
    num_pixel: int = 16,
    seed: int = 0,
    **kwargs,
):
    """
    Return a small random image-classification dataset for one client.

    The data is a pure function of ``(seed, client_id)``, so every process that
    builds the same client gets the same shard, with nothing to download.

    :param num_clients: total number of clients (unused; kept for the loader API)
    :param client_id: the client id
    :return: (train_dataset, val_dataset)
    """
    train = torch.Generator().manual_seed(seed * 1000 + client_id)
    val = torch.Generator().manual_seed(seed * 1000 + 999)
    shape = (1, num_pixel, num_pixel)
    train_dataset = Dataset(
        torch.rand(num_samples, *shape, generator=train),
        torch.randint(0, num_classes, (num_samples,), generator=train),
    )
    val_dataset = Dataset(
        torch.rand(16, *shape, generator=val),
        torch.randint(0, num_classes, (16,), generator=val),
    )
    return train_dataset, val_dataset
