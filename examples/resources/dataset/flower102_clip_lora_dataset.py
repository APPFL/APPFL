"""
Flower102 dataset loader for decentralized CLIP + LoRA experiments.

Uses clip_lora's dataset infrastructure (datasets.build_dataset) to load the
Oxford Flowers-102 dataset with CLIP preprocessing.  The returned dataset
objects carry two extra attributes read by CLIPLoRATrainer:

    .classnames (list[str]): 102 human-readable flower class names.
    .template   (list[str]): text template(s), e.g. ["a photo of a {}."]

Reads the Zhou-split (split_zhou_OxfordFlowers.json) via clip_lora's own
Flowers-102 dataset class, so the few-shot ``shots`` sampling is identical to
the non-federated clip_lora baseline.

IID partitioning:
    Each class is divided evenly across all clients.  Every client gets a
    proportional share of every class (class-balanced).

Non-IID partitioning:
    Classes are assigned exclusively to individual clients via round-robin.

Args to get_flower102_clip_lora():
    data_path      (str) : path to the Flower102 root directory.
    backbone       (str) : CLIP backbone string, e.g. "ViT-B/16".
    shots          (int) : few-shot samples per class (default 16).
    num_clients    (int) : total number of FL clients.
    client_id      (int) : 0-based index of this client.
    data_dist      (str) : "iid" (default) or "non-iid".
    seed           (int) : random seed for partitioning (default 1).
    clip_lora_root (str) : absolute path to the clip_lora project directory.

Returns:
    (train_dataset, val_dataset) — both are _CLIPDatasetWrapper instances
    that expose .classnames and .template.
"""

import os
import sys
import random
from collections import defaultdict
from typing import Optional

from torch.utils.data import Dataset
from PIL import Image


# ---------------------------------------------------------------------------
# Internal dataset wrapper
# ---------------------------------------------------------------------------


class _CLIPDatasetWrapper(Dataset):
    """
    Wraps a list of clip_lora Datum objects into a standard
    (image, label) torch Dataset with CLIP preprocessing.

    Extra attributes read by CLIPLoRATrainer:
        classnames (list[str]): human-readable class labels.
        template   (list[str]): text template strings.
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


# ---------------------------------------------------------------------------
# Public factory function
# ---------------------------------------------------------------------------


def get_flower102_clip_lora(
    data_path: str,
    backbone: str = "ViT-B/16",
    shots: int = 16,
    num_clients: int = 5,
    client_id: int = 0,
    data_dist: str = "iid",
    seed: int = 1,
    clip_lora_root: Optional[str] = None,
    **kwargs,
):
    """
    Return (train_dataset, val_dataset) for the given Flower102 client.

    Parameters
    ----------
    data_path      : Root directory of the Flower102 dataset.
    backbone       : CLIP backbone (must match model_kwargs.backbone).
    shots          : Few-shot samples per class.
    num_clients    : Total number of federated clients.
    client_id      : Zero-based index of this client.
    data_dist      : "iid" or "non-iid".
    seed           : Random seed for partitioning.
    clip_lora_root : Path to the clip_lora project directory.

    Returns
    -------
    (train_subset, val_dataset) : both are _CLIPDatasetWrapper instances.
    """
    _add_clip_lora_to_path(clip_lora_root)

    import clip
    from clip_benchmarks import build_dataset
    import torchvision.transforms as transforms

    # CLIP training augmentation
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=224,
                scale=(0.08, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    # CLIP val/test preprocessing
    _, clip_preprocess = clip.load(backbone, device="cpu")

    # Build the full Flowers-102 dataset via clip_lora infrastructure
    # dataset_name "oxford_flowers" maps to Flowers102 in clip_lora's registry
    full_dataset = build_dataset("oxford_flowers", data_path, shots, clip_preprocess)

    classnames = full_dataset.classnames
    template = full_dataset.template

    # Partition training data for this client
    train_x = full_dataset.train_x  # list of Datum objects
    client_data = _partition(train_x, num_clients, client_id, data_dist, seed)

    train_dataset = _CLIPDatasetWrapper(
        client_data, train_transform, classnames, template
    )
    val_dataset = _CLIPDatasetWrapper(
        full_dataset.val, clip_preprocess, classnames, template
    )

    return train_dataset, val_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add_clip_lora_to_path(clip_lora_root: Optional[str]):
    """Insert clip_lora into sys.path so its packages are importable."""
    if clip_lora_root is not None:
        if clip_lora_root not in sys.path:
            sys.path.insert(0, clip_lora_root)
        return

    # Auto-detect based on expected repo layout relative to this file
    candidate = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "clip_lora")
    )
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.insert(0, candidate)


def _partition(
    data_source: list,
    num_clients: int,
    client_id: int,
    data_dist: str,
    seed: int,
) -> list:
    """
    Partition data_source (list of Datum) for a single client.

    IID:     Each class is divided evenly across all clients (class-balanced).
    Non-IID: Classes are assigned exclusively to individual clients
             via round-robin over a shuffled class list.
    """
    rng = random.Random(seed)

    class_to_data = defaultdict(list)
    for item in data_source:
        class_to_data[item.label].append(item)

    if data_dist == "iid":
        client_data = []
        for cls_items in class_to_data.values():
            shuffled = cls_items[:]
            rng.shuffle(shuffled)
            n = len(shuffled)
            base = n // num_clients
            remainder = n % num_clients
            start = client_id * base + min(client_id, remainder)
            end = start + base + (1 if client_id < remainder else 0)
            client_data.extend(shuffled[start:end])
        return client_data

    elif data_dist == "non-iid":
        classes = list(class_to_data.keys())
        rng.shuffle(classes)
        client_data = []
        for idx, cls in enumerate(classes):
            if idx % num_clients == client_id:
                client_data.extend(class_to_data[cls])
        return client_data

    else:
        raise ValueError(f"data_dist must be 'iid' or 'non-iid', got '{data_dist}'")
