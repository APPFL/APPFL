"""Regression test for the divide-by-zero NaN crash in Dirichlet partitioning.

Before the fix, ``dirichlet_noniid_partition`` (and its DataFrame variant) divided
the per-class normalization by ``np.dot(weights, individuals[:, j])`` with no
guard. For strongly non-IID settings (many classes + many clients + small
Dirichlet ``alpha2``) a whole class column of ``individuals`` can be exactly zero,
making the denominator 0 -> 0/0 = NaN -> ``int(sample_matrix[...])`` raised
``ValueError: cannot convert float NaN to integer``.

The parameters below (100 classes, 128 clients, alpha2=0.01, seed=17) were found
to produce an all-zero class column, i.e. they crash on the unfixed code and pass
with the guard. This test therefore fails if the guard is ever removed.
"""

import torch
from torch.utils.data import Dataset

from appfl.misc.data import dirichlet_noniid_partition


class _TinyLabeled(Dataset):
    """Minimal (input, label) dataset with a controllable number of classes."""

    def __init__(self, num_classes: int, per_class: int):
        labels = []
        for c in range(num_classes):
            labels.extend([c] * per_class)
        self._x = torch.zeros((len(labels), 1, 2, 2), dtype=torch.float32)
        self._y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self._y)

    def __getitem__(self, idx):
        return self._x[idx], int(self._y[idx])


def test_dirichlet_partition_zero_class_column_no_nan_crash():
    """A settings combo that yields an all-zero class column must not crash.

    On the unfixed code this raises
    ``ValueError: cannot convert float NaN to integer``.
    """
    num_clients = 128
    dataset = _TinyLabeled(num_classes=100, per_class=10)

    client_datasets = dirichlet_noniid_partition(
        dataset,
        num_clients,
        visualization=False,
        alpha1=num_clients,
        alpha2=0.01,
        seed=17,
    )

    # Must return one dataset per client with a valid integer sample count, and
    # every sample must be accounted for exactly once.
    assert len(client_datasets) == num_clients
    total = 0
    for ds in client_datasets:
        n = len(ds)
        assert isinstance(n, int)
        assert n >= 0
        total += n
    assert total == len(dataset)
