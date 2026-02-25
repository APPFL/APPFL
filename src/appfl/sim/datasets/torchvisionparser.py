from __future__ import annotations

import inspect
import logging
from pathlib import Path

from appfl.sim.datasets.common import (
    clientize_raw_dataset,
    finalize_dataset_outputs,
    infer_num_classes,
    make_load_tag,
    resolve_dataset_logger,
    to_namespace,
)

try:
    from torchvision import datasets as tv_datasets
    from torchvision import transforms
except Exception:  # pragma: no cover
    tv_datasets = None
    transforms = None


logger = logging.getLogger(__name__)


def _resolve_torchvision_dataset(name: str):
    if tv_datasets is None:
        return None
    if hasattr(tv_datasets, name):
        return getattr(tv_datasets, name)
    for candidate in dir(tv_datasets):
        if candidate.lower() == name.lower():
            return getattr(tv_datasets, candidate)
    return None


def _ensure_targets(ds):
    if hasattr(ds, "targets"):
        return
    for attr in ["labels", "_labels", "y"]:
        if hasattr(ds, attr):
            setattr(ds, "targets", getattr(ds, attr))
            return
    if hasattr(ds, "_samples"):
        setattr(ds, "targets", [s[-1] for s in ds._samples])


_MNIST_S3_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
_MNIST_FAMILY = {"MNIST", "FashionMNIST", "KMNIST", "EMNIST", "QMNIST"}


def _patch_mnist_mirrors(dataset_cls) -> None:
    """Replace mirrors with only the reliable S3 source for MNIST-family datasets.

    yann.lecun.com returns a corrupt response that passes URL fetching but fails
    MD5 verification, and torchvision does not fall back on that error. Replacing
    the list entirely ensures the S3 mirror is always used.
    """
    if dataset_cls.__name__ in _MNIST_FAMILY and hasattr(dataset_cls, "mirrors"):
        dataset_cls.mirrors = [_MNIST_S3_MIRROR]


def _instantiate_dataset(dataset_cls, root: str, split: str, download: bool, transform):
    sig = inspect.signature(dataset_cls.__init__)
    kwargs = {}
    if "root" in sig.parameters:
        kwargs["root"] = root
    if "download" in sig.parameters:
        kwargs["download"] = download
    if "transform" in sig.parameters:
        kwargs["transform"] = transform

    if "train" in sig.parameters:
        kwargs["train"] = split == "train"
        return dataset_cls(**kwargs)

    if "split" in sig.parameters:
        candidates = (
            ["train", "training", "trainval"]
            if split == "train"
            else ["test", "valid", "val"]
        )
        for cand in candidates:
            try:
                return dataset_cls(split=cand, **kwargs)
            except Exception:
                continue

    return dataset_cls(**kwargs)


def fetch_torchvision_dataset(args):
    args = to_namespace(args)
    active_logger = resolve_dataset_logger(args, logger)
    tag = make_load_tag(str(args.dataset_name), benchmark="TORCHVISION")
    active_logger.info("[%s] resolving dataset class.", tag)
    if tv_datasets is None:
        raise RuntimeError("torchvision is not installed.")

    dataset_cls = _resolve_torchvision_dataset(args.dataset_name)
    if dataset_cls is None:
        raise ValueError(f"Unknown torchvision dataset: {args.dataset_name}")

    _patch_mnist_mirrors(dataset_cls)
    data_root = Path(str(args.data_dir)).expanduser()
    data_root.mkdir(parents=True, exist_ok=True)
    if bool(args.download):
        active_logger.info("[%s] downloading (if needed).", tag)

    transform = transforms.ToTensor() if transforms is not None else None
    raw_train = _instantiate_dataset(
        dataset_cls,
        str(data_root),
        "train",
        bool(args.download),
        transform,
    )
    raw_test = _instantiate_dataset(
        dataset_cls,
        str(data_root),
        "test",
        bool(args.download),
        transform,
    )

    _ensure_targets(raw_train)
    _ensure_targets(raw_test)
    active_logger.info("[%s] building federated client splits.", tag)

    client_datasets = clientize_raw_dataset(raw_train, args)
    client_datasets, server_dataset, dataset_meta = finalize_dataset_outputs(
        client_datasets=client_datasets,
        server_dataset=raw_test,
        dataset_meta=args,
        raw_train=raw_train,
    )
    dataset_meta.need_embedding = False
    dataset_meta.seq_len = None
    dataset_meta.num_embeddings = None
    if hasattr(raw_train, "classes"):
        dataset_meta.num_classes = max(
            int(dataset_meta.num_classes), int(len(raw_train.classes))
        )
    else:
        dataset_meta.num_classes = int(infer_num_classes(raw_train))
    active_logger.info(
        "[%s] finished loading (%d clients).", tag, int(dataset_meta.num_clients)
    )
    return client_datasets, server_dataset, dataset_meta
