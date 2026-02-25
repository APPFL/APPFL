from __future__ import annotations

from typing import Any

from appfl.sim.datasets import (
    fetch_custom_dataset,
    fetch_flamby,
    fetch_hf_dataset,
    fetch_leaf,
    fetch_medmnist_dataset,
    fetch_tff_dataset,
    fetch_torchaudio_dataset,
    fetch_torchtext_dataset,
    fetch_torchvision_dataset,
)
from appfl.sim.datasets.common import to_namespace


LEAF_DATASETS = {"FEMNIST", "SHAKESPEARE", "SENT140", "CELEBA", "REDDIT"}
FLAMBY_DATASET_KEYS = {"HEART", "ISIC2019", "IXITINY"}


def _has_medmnist_dataset(name: str) -> bool:
    try:
        import medmnist

        return str(name).lower() in {k.lower() for k in medmnist.INFO.keys()}
    except Exception:
        return False


def _has_torchtext_dataset(name: str) -> bool:
    try:
        import torchtext

        return hasattr(torchtext.datasets, str(name))
    except Exception:
        return False


def _has_torchaudio_dataset(name: str) -> bool:
    try:
        import torchaudio

        if hasattr(torchaudio.datasets, str(name)):
            return True
        lowered = str(name).lower()
        return any(c.lower() == lowered for c in dir(torchaudio.datasets))
    except Exception:
        return False


def load_dataset(args: Any):
    """Unified dataset loader API.

    Return contract:
      client_datasets, server_dataset, dataset_meta

    `dataset.backend` modes:
    - `auto`: infer parser by dataset name/library.
    - `custom`: local path or callable parser (`dataset.path` / `dataset.configs.entrypoint`).
    - `hf`: first-class HuggingFace dataset backend (supports plain repo id in `dataset.name`).
    - built-ins: `torchvision`, `torchtext`, `torchaudio`, `medmnist`, `flamby`, `leaf`, `tff`.
    """
    args = to_namespace(args)
    mode = str(getattr(args, "dataset_backend", "torchvision")).strip().lower()
    dataset_name = str(args.dataset_name)
    dataset_upper = dataset_name.upper()
    dataset_lower = dataset_name.lower()

    if mode in {"custom"}:
        return fetch_custom_dataset(args)
    if mode == "hf":
        return fetch_hf_dataset(args)
    if mode == "torchvision":
        return fetch_torchvision_dataset(args)
    if mode == "torchtext":
        return fetch_torchtext_dataset(args)
    if mode == "torchaudio":
        return fetch_torchaudio_dataset(args)
    if mode == "medmnist":
        return fetch_medmnist_dataset(args)
    if mode == "flamby":
        return fetch_flamby(args)
    if mode == "leaf":
        return fetch_leaf(args)
    if mode == "tff":
        return fetch_tff_dataset(args)
    if mode != "auto":
        raise ValueError(
            "dataset.backend must be one of: auto, custom, hf, "
            "torchvision, torchtext, torchaudio, medmnist, flamby, leaf, tff"
        )

    # auto mode
    if dataset_lower.startswith("hf:"):
        return fetch_hf_dataset(args)
    if dataset_upper in LEAF_DATASETS:
        return fetch_leaf(args)
    if dataset_upper in FLAMBY_DATASET_KEYS:
        return fetch_flamby(args)
    if dataset_lower.startswith("tff:"):
        return fetch_tff_dataset(args)

    if _has_medmnist_dataset(dataset_name):
        return fetch_medmnist_dataset(args)
    if _has_torchtext_dataset(dataset_name):
        return fetch_torchtext_dataset(args)
    if _has_torchaudio_dataset(dataset_name):
        return fetch_torchaudio_dataset(args)

    return fetch_torchvision_dataset(args)
