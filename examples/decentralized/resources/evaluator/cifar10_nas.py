"""The CIFAR-10 NAS oracle -- ADKO Algorithm 1 step 9 for the CIFAR-10 study.

Evaluating a design point means building the ResNet it describes, training it on this agent's
private shard, and returning validation accuracy in ``[0, 1]``. Mirrors ``cifar10/model.py`` and
``cifar10/data.py`` in the reference implementation (github.com/lucasrillo/adko): the same
CIFAR-adapted ResNet, Adam with an optional cosine schedule, no augmentation, and the same
shard construction.

This is where the agents differ. They all search the same space; what each one owns is its
data. ``non_iid`` gives agent ``i`` a shard that is 80% its own block of ``10 / num_agents``
classes and 20% the rest -- disjoint across agents, so an architecture tuned on one agent's
classes is evidence, not proof, for another's. ``iid`` gives every agent a uniform sample.

Shards are computed from the training labels alone, replaying the reference's random draws, so
for the same seed agent ``i`` gets exactly the images it gets in the reference -- and only agent
``i``'s images are converted to tensors.

Needs ``torch`` and ``torchvision``. CIFAR-10 is downloaded to ``data_dir`` on first use; on a
cluster whose compute nodes have no internet, run the download once on a login node.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_SPACE_DIR = Path(__file__).resolve().parents[1] / "space"
if str(_SPACE_DIR) not in sys.path:
    sys.path.insert(0, str(_SPACE_DIR))

from cifar10_nas_space import decode_theta, readable_theta  # noqa: E402

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
NUM_CLASSES = 10

# -- data ---------------------------------------------------------------------------------

#: The decoded training set, once per process: every agent in a serial run reads the same one.
_TRAINSET: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _load_trainset(data_dir: str, download: bool) -> tuple[np.ndarray, np.ndarray]:
    """``(images uint8 [N,32,32,3], labels int [N])`` for the CIFAR-10 training split."""
    if data_dir not in _TRAINSET:
        import torchvision

        ds = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=download)
        _TRAINSET[data_dir] = (np.asarray(ds.data), np.asarray(ds.targets))
    return _TRAINSET[data_dir]


def noniid_split(
    targets: np.ndarray,
    n_agents: int,
    shard_size: int | None,
    primary_frac: float,
    val_frac: float,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """``[(train_idx, val_idx)]`` per agent, as ``data.py::create_shards`` draws them.

    Classes are split into ``n_agents`` contiguous blocks; agent ``i`` takes ``primary_frac`` of
    its shard from its own block and the rest evenly from every other class. Each class's
    indices are permuted once and consumed from the front, so the shards are disjoint.
    """
    rng = np.random.RandomState(seed)
    num_classes = int(targets.max()) + 1
    if n_agents < 2 or num_classes % n_agents != 0:
        raise ValueError(
            f"non_iid gives each agent {num_classes} / num_agents classes, so num_agents must "
            f"divide {num_classes} and be at least 2, not {n_agents}. With CIFAR-10 that is 2 "
            f"or 5 (10 cannot fit the default shard_size). Use distribution: iid for any count."
        )
    if shard_size is None:
        shard_size = len(targets) // n_agents

    n_primary = num_classes // n_agents
    n_other = num_classes - n_primary
    per_primary = int(round(primary_frac * shard_size / n_primary))
    per_other = int(round((1 - primary_frac) * shard_size / n_other))
    needed = per_primary + (n_agents - 1) * per_other
    available = min(int((targets == c).sum()) for c in range(num_classes))
    if needed > available:
        raise ValueError(
            f"the partition needs {needed} images per class but CIFAR-10 has {available}; "
            f"reduce shard_size"
        )

    pools = [rng.permutation(np.where(targets == c)[0]) for c in range(num_classes)]
    taken = [0] * num_classes

    def take(c: int, k: int) -> np.ndarray:
        out = pools[c][taken[c] : taken[c] + k]
        taken[c] += k
        return out

    splits = []
    for i in range(n_agents):
        primary = set(range(i * n_primary, (i + 1) * n_primary))
        parts = [take(c, per_primary) for c in sorted(primary)]
        parts += [take(c, per_other) for c in range(num_classes) if c not in primary]
        idx = np.concatenate(parts)
        rng.shuffle(idx)
        n_val = int(val_frac * len(idx))
        splits.append((idx[n_val:], idx[:n_val]))
    return splits


def iid_split(
    n_total: int, n_agents: int, shard_size: int, val_frac: float, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """``[(train_idx, val_idx)]`` per agent, as ``data.py::create_iid_shards`` draws them:
    an independent uniform sample per agent, so shards may overlap."""
    rng = np.random.RandomState(seed)
    splits = []
    for _ in range(n_agents):
        idx = rng.choice(n_total, shard_size, replace=False)
        rng.shuffle(idx)
        n_val = int(val_frac * len(idx))
        splits.append((idx[n_val:], idx[:n_val]))
    return splits


def _to_tensors(
    images: np.ndarray, labels: np.ndarray, idx: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    """``ToTensor()`` then ``Normalize(mean, std)``, applied to ``idx`` only."""
    x = torch.from_numpy(images[idx]).permute(0, 3, 1, 2).float().div(255)
    mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
    return ((x - mean) / std).contiguous(), torch.from_numpy(labels[idx]).long()


# -- model --------------------------------------------------------------------------------


class BasicBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class CIFARResNet(nn.Module):
    """3x3 stem, no max-pool, three groups of ``blocks_per_group`` blocks at widths w, 2w, 4w."""

    def __init__(
        self,
        blocks_per_group: int = 3,
        base_width: int = 16,
        dropout: float = 0.0,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        w = base_width
        self.in_planes = w
        self.conv1 = nn.Conv2d(3, w, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(w)
        self.layer1 = self._make_layer(w, blocks_per_group, stride=1)
        self.layer2 = self._make_layer(w * 2, blocks_per_group, stride=2)
        self.layer3 = self._make_layer(w * 4, blocks_per_group, stride=2)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(w * 4, num_classes)

    def _make_layer(self, planes: int, n_blocks: int, stride: int) -> nn.Sequential:
        layers = []
        for s in [stride] + [1] * (n_blocks - 1):
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer3(self.layer2(self.layer1(out)))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(self.dropout(out))


def _train(model, loader, params: dict, n_epochs: int, device) -> None:
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["lr"], weight_decay=params["wd"]
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        if params["cosine_schedule"]
        else None
    )
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(n_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()


@torch.inference_mode()
def _accuracy(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        correct += (model(xb).argmax(1) == yb).sum().item()
        total += len(yb)
    return correct / total if total else 0.0


# -- entry point --------------------------------------------------------------------------


def _resolve_device(device: str, agent_index: int) -> torch.device:
    """``auto`` spreads agents over the visible GPUs; under MPI with one GPU per rank that is
    simply ``cuda:0`` in every process."""
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{agent_index % torch.cuda.device_count()}")
    return torch.device("cpu")


def get_evaluator(
    agent_index: int = 0,
    num_agents: int = 5,
    distribution: str = "non_iid",
    data_dir: str = "./datasets/RawData",
    download: bool = True,
    n_epochs: int = 40,
    shard_size: int = 10000,
    primary_frac: float = 0.8,
    val_frac: float = 0.2,
    data_seed: int = 42,
    device: str = "auto",
) -> Callable[[Sequence[float]], float]:
    """Entry point named by ``evaluator_configs.evaluator_name``.

    ``agent_index`` and ``num_agents`` select this agent's shard. The configs take them from
    ``space_configs.space_kwargs`` by interpolation, which is where every launcher writes them.
    """
    if distribution not in ("non_iid", "iid"):
        raise ValueError(
            f"distribution must be 'non_iid' or 'iid', not {distribution!r}"
        )
    if not 0 <= agent_index < num_agents:
        raise ValueError(f"agent_index {agent_index} outside 0..{num_agents - 1}")

    images, labels = _load_trainset(data_dir, download)
    if distribution == "non_iid":
        splits = noniid_split(
            labels, num_agents, shard_size, primary_frac, val_frac, data_seed
        )
    else:
        splits = iid_split(len(labels), num_agents, shard_size, val_frac, data_seed)
    train_idx, val_idx = splits[agent_index]
    X_train, y_train = _to_tensors(images, labels, train_idx)
    X_val, y_val = _to_tensors(images, labels, val_idx)
    dev = _resolve_device(device, agent_index)

    def evaluate(point: Sequence[float]) -> float:
        params = decode_theta(point)
        try:
            model = CIFARResNet(
                blocks_per_group=params["blocks_per_group"],
                base_width=params["base_width"],
                dropout=params["dropout"],
            ).to(dev)
            train_dl = DataLoader(
                TensorDataset(X_train, y_train),
                batch_size=params["batch_size"],
                shuffle=True,
            )
            val_dl = DataLoader(
                TensorDataset(X_val, y_val), batch_size=params["batch_size"]
            )
            _train(model, train_dl, params, n_epochs, dev)
            return _accuracy(model, val_dl, dev)
        except RuntimeError as exc:
            # The reference scores a configuration that cannot be trained (CUDA out of
            # memory, typically) as 0 rather than aborting the run. Narrower than its bare
            # `except Exception`, so a programming error still surfaces.
            print(
                f"[cifar10_nas] agent {agent_index}: {readable_theta(point)} failed: {exc}"
            )
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            return 0.0

    return evaluate
