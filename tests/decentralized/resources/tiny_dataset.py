import torch
from appfl.misc.data import Dataset


def get_tiny(num_clients: int = 2, client_id: int = 0, num_samples: int = 16, **kwargs):
    """A deterministic synthetic dataset, so tests neither download nor depend on the network.

    Each client gets a different slice, seeded from its id, which is enough for the nodes'
    local models to actually diverge between rounds -- without that, an aggregation bug would
    be invisible because every node would already agree.
    """
    generator = torch.Generator().manual_seed(1234 + client_id)
    inputs = torch.randn(num_samples, 4, generator=generator)
    labels = (inputs.sum(dim=1) > 0).long()
    return Dataset(inputs, labels), Dataset(inputs, labels)
