"""
General dataset loader for federated CLIP + LoRA (DeCaF) experiments.

Supports two dataset backends behind a single entry point:

  1. CLIP benchmark datasets (via clip_benchmarks in resources/clip_lora/):
       oxford_flowers, oxford_pets, caltech101, dtd, eurosat, fgvc,
       food101, stanford_cars, sun397, ucf101, imagenet

  2. General torchvision datasets (downloaded automatically):
       cifar10, cifar100, mnist, stl10

Both backends return (train_dataset, val_dataset) where each dataset
carries the extra attributes DeCaFTrainer needs:
    .classnames (list[str]): human-readable class labels.
    .template   (list[str]): text prompt template(s).

Both backends support:
    shots    – few-shot sampling (N examples per class); 0/None = full dataset.
    data_dist – "iid"     : class-balanced split across clients.
               "non-iid" : classes assigned exclusively to clients.

Usage in client_decaf_fedavg.yaml:
    data_configs:
      dataset_path: "./resources/dataset/decaf_dataset.py"
      dataset_name: "get_decaf_dataset"
      dataset_kwargs:
        dataset_name: "cifar10"      # any supported dataset name
        root_path:    "/path/to/data"
        shots:        16             # few-shot per class; 0 or null = full
        backbone:     "ViT-B/16"     # must match model_kwargs.backbone
        data_dist:    "iid"          # "iid" or "non-iid"
        num_clients:  5              # overridden by MPI script
        client_id:    0              # overridden by MPI script
        seed:         1
"""

import os
import random
import sys
from collections import defaultdict
from typing import List, Optional

import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Dataset backend registry
# ---------------------------------------------------------------------------

# Datasets served by clip_benchmarks (resources/clip_lora/clip_benchmarks/)
_CLIP_BENCHMARK_DATASETS = {
    "oxford_flowers",
    "oxford_pets",
    "caltech101",
    "dtd",
    "eurosat",
    "fgvc",
    "food101",
    "stanford_cars",
    "sun397",
    "ucf101",
    "imagenet",
}

# Torchvision-backed datasets.
# classnames=None means we pull them from dataset.classes at load time.
_TORCHVISION_REGISTRY = {
    "cifar10": {
        "loader": torchvision.datasets.CIFAR10,
        "classnames": [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ],
        "template": ["a photo of a {}."],
        "split_kwargs": lambda train: {"train": train},
    },
    "cifar100": {
        "loader": torchvision.datasets.CIFAR100,
        "classnames": None,  # pulled from dataset.classes
        "template": ["a photo of a {}."],
        "split_kwargs": lambda train: {"train": train},
    },
    "mnist": {
        "loader": torchvision.datasets.MNIST,
        "classnames": [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
        ],
        "template": ["a handwritten digit {}."],
        "split_kwargs": lambda train: {"train": train},
        "grayscale": True,  # needs L→RGB conversion
    },
    "stl10": {
        "loader": torchvision.datasets.STL10,
        "classnames": None,
        "template": ["a photo of a {}."],
        "split_kwargs": lambda train: {"split": "train" if train else "test"},
    },
}


# ---------------------------------------------------------------------------
# Dataset wrappers
# ---------------------------------------------------------------------------


class _CLIPBenchmarkWrapper(Dataset):
    """
    Wraps a list of clip_benchmarks Datum objects into a torch Dataset.
    Each Datum has .impath (file path) and .label (int).
    """

    def __init__(self, data_source, transform, classnames, template):
        self._data = data_source
        self._transform = transform
        self.classnames = classnames
        self.template = template

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        datum = self._data[idx]
        img = Image.open(datum.impath).convert("RGB")
        if self._transform is not None:
            img = self._transform(img)
        return img, datum.label


class _TorchvisionWrapper(Dataset):
    """
    Wraps a torchvision dataset (loaded without transform) restricted
    to a given list of indices, applying a CLIP transform per item.
    """

    def __init__(self, base_dataset, indices, transform, classnames, template):
        self._dataset = base_dataset
        self._indices = indices
        self._transform = transform
        self.classnames = classnames
        self.template = template

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        img, label = self._dataset[self._indices[idx]]
        # img is a PIL Image when base_dataset was loaded without transform
        if self._transform is not None:
            img = self._transform(img)
        return img, int(label)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def get_decaf_dataset(
    dataset_name: str,
    root_path: str,
    num_clients: int,
    client_id: int,
    shots: int = 0,
    data_dist: str = "iid",
    backbone: str = "ViT-B/16",
    seed: int = 1,
    clip_lora_root: Optional[str] = None,
    **kwargs,
):
    """
    Return (train_dataset, val_dataset) for a single federated client.

    Args:
        dataset_name:   Dataset identifier (see supported names above).
        root_path:      Root directory for raw data files.
        num_clients:    Total FL clients (set by MPI script).
        client_id:      Zero-based client index (set by MPI script).
        shots:          Few-shot examples per class. 0 / None = full dataset.
        data_dist:      "iid" or "non-iid".
        backbone:       CLIP backbone name for preprocessing (e.g. "ViT-B/16").
        seed:           RNG seed for partitioning / few-shot sampling.
        clip_lora_root: Optional override for clip_lora root path.

    Returns:
        (train_dataset, val_dataset)
    """
    _add_clip_lora_to_path(clip_lora_root)
    import clip

    _, clip_preprocess = clip.load(backbone, device="cpu")

    if dataset_name in _CLIP_BENCHMARK_DATASETS:
        return _load_clip_benchmark(
            dataset_name,
            root_path,
            shots,
            clip_preprocess,
            num_clients,
            client_id,
            data_dist,
            seed,
        )
    elif dataset_name in _TORCHVISION_REGISTRY:
        return _load_torchvision(
            dataset_name,
            root_path,
            shots,
            clip_preprocess,
            num_clients,
            client_id,
            data_dist,
            seed,
        )
    else:
        supported = sorted(_CLIP_BENCHMARK_DATASETS | set(_TORCHVISION_REGISTRY))
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Supported datasets: {supported}"
        )


# ---------------------------------------------------------------------------
# Backend: CLIP benchmark datasets
# ---------------------------------------------------------------------------


def _load_clip_benchmark(
    dataset_name,
    root_path,
    shots,
    clip_preprocess,
    num_clients,
    client_id,
    data_dist,
    seed,
):
    from clip_benchmarks import build_dataset

    train_transform = _clip_train_transform()
    full_dataset = build_dataset(dataset_name, root_path, shots or 0, clip_preprocess)

    classnames = full_dataset.classnames
    template = full_dataset.template
    train_x = full_dataset.train_x  # list of Datum objects

    client_data = _partition_datums(train_x, num_clients, client_id, data_dist, seed)

    train_dataset = _CLIPBenchmarkWrapper(
        client_data, train_transform, classnames, template
    )
    val_dataset = _CLIPBenchmarkWrapper(
        full_dataset.val, clip_preprocess, classnames, template
    )
    return train_dataset, val_dataset


# ---------------------------------------------------------------------------
# Backend: Torchvision datasets
# ---------------------------------------------------------------------------


def _load_torchvision(
    dataset_name,
    root_path,
    shots,
    clip_preprocess,
    num_clients,
    client_id,
    data_dist,
    seed,
):
    cfg = _TORCHVISION_REGISTRY[dataset_name]
    loader_cls = cfg["loader"]
    split_kwargs = cfg["split_kwargs"]
    grayscale = cfg.get("grayscale", False)

    # Build transforms (no transform on base dataset; applied in wrapper)
    train_transform = _clip_train_transform(grayscale=grayscale)
    val_transform = clip_preprocess
    if grayscale:
        _to_rgb = T.Lambda(lambda img: img.convert("RGB"))
        val_transform = T.Compose([_to_rgb, clip_preprocess])

    # Load base datasets without transform
    train_base = loader_cls(
        root_path, download=True, transform=None, **split_kwargs(True)
    )
    val_base = loader_cls(
        root_path, download=True, transform=None, **split_kwargs(False)
    )

    # Resolve classnames
    classnames = cfg["classnames"]
    if classnames is None:
        classnames = [c.replace("_", " ") for c in train_base.classes]
    template = cfg["template"]

    # Few-shot sampling on train split
    all_train_indices = _few_shot_sample(train_base, shots, seed)

    # Partition among clients
    client_indices = _partition_indices(
        train_base, all_train_indices, num_clients, client_id, data_dist, seed
    )

    train_dataset = _TorchvisionWrapper(
        train_base, client_indices, train_transform, classnames, template
    )
    val_dataset = _TorchvisionWrapper(
        val_base, list(range(len(val_base))), val_transform, classnames, template
    )
    return train_dataset, val_dataset


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _clip_train_transform(grayscale: bool = False):
    """Standard CLIP training augmentation."""
    steps = []
    if grayscale:
        steps.append(T.Lambda(lambda img: img.convert("RGB")))
    steps += [
        T.RandomResizedCrop(
            size=224,
            scale=(0.08, 1.0),
            interpolation=T.InterpolationMode.BICUBIC,
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ]
    return T.Compose(steps)


def _get_targets(dataset) -> list:
    """Extract integer labels from a torchvision dataset."""
    if hasattr(dataset, "targets"):
        t = dataset.targets
        return t.tolist() if isinstance(t, torch.Tensor) else list(t)
    if hasattr(dataset, "labels"):
        t = dataset.labels
        return t.tolist() if isinstance(t, torch.Tensor) else list(t)
    return [int(label) for _, label in dataset]


def _few_shot_sample(dataset, shots: int, seed: int) -> List[int]:
    """
    Return a flat list of dataset indices after few-shot sampling.
    If shots <= 0 or None, returns all indices.
    """
    n = len(dataset)
    if not shots or shots <= 0:
        return list(range(n))

    targets = _get_targets(dataset)
    rng = random.Random(seed)

    class_to_idx: dict = defaultdict(list)
    for i, label in enumerate(targets):
        class_to_idx[label].append(i)

    selected = []
    for cls_indices in class_to_idx.values():
        shuffled = cls_indices[:]
        rng.shuffle(shuffled)
        selected.extend(shuffled[:shots])
    return selected


def _partition_indices(
    dataset,
    indices: List[int],
    num_clients: int,
    client_id: int,
    data_dist: str,
    seed: int,
) -> List[int]:
    """
    Partition a list of dataset indices for a single client.

    IID:     Class-balanced split across all clients.
    Non-IID: Classes assigned exclusively (round-robin).
    """
    targets = _get_targets(dataset)
    rng = random.Random(seed)

    class_to_idx: dict = defaultdict(list)
    for i in indices:
        class_to_idx[targets[i]].append(i)

    if data_dist == "iid":
        client_data = []
        for cls_indices in class_to_idx.values():
            shuffled = cls_indices[:]
            rng.shuffle(shuffled)
            n = len(shuffled)
            base, remainder = divmod(n, num_clients)
            start = client_id * base + min(client_id, remainder)
            end = start + base + (1 if client_id < remainder else 0)
            client_data.extend(shuffled[start:end])
        return client_data

    elif data_dist == "non-iid":
        classes = list(class_to_idx.keys())
        rng.shuffle(classes)
        return [
            idx
            for pos, cls in enumerate(classes)
            if pos % num_clients == client_id
            for idx in class_to_idx[cls]
        ]

    else:
        raise ValueError(f"data_dist must be 'iid' or 'non-iid', got '{data_dist}'")


def _partition_datums(
    data_source: list,
    num_clients: int,
    client_id: int,
    data_dist: str,
    seed: int,
) -> list:
    """
    Partition a list of clip_benchmarks Datum objects for a single client.
    Mirrors _partition_indices but for Datum objects (.label attribute).
    """
    rng = random.Random(seed)

    class_to_data: dict = defaultdict(list)
    for item in data_source:
        class_to_data[item.label].append(item)

    if data_dist == "iid":
        client_data = []
        for cls_items in class_to_data.values():
            shuffled = cls_items[:]
            rng.shuffle(shuffled)
            n = len(shuffled)
            base, remainder = divmod(n, num_clients)
            start = client_id * base + min(client_id, remainder)
            end = start + base + (1 if client_id < remainder else 0)
            client_data.extend(shuffled[start:end])
        return client_data

    elif data_dist == "non-iid":
        classes = list(class_to_data.keys())
        rng.shuffle(classes)
        return [
            item
            for pos, cls in enumerate(classes)
            if pos % num_clients == client_id
            for item in class_to_data[cls]
        ]

    else:
        raise ValueError(f"data_dist must be 'iid' or 'non-iid', got '{data_dist}'")


def _add_clip_lora_to_path(clip_lora_root: Optional[str]):
    """Add clip_lora packages to sys.path so clip and clip_benchmarks are importable."""
    _bundled = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "clip_lora")
    )
    for _p in filter(None, [_bundled, clip_lora_root]):
        if os.path.isdir(_p) and _p not in sys.path:
            sys.path.insert(0, _p)
