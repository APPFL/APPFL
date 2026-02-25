from __future__ import annotations

import inspect
import logging
from typing import Any

from appfl.sim.datasets.common import (
    make_load_tag,
    resolve_dataset_logger,
    resolve_fixed_pool_clients,
    to_namespace,
)


# Keep FLamby dataset scope aligned with AAggFF parser support.
_SUPPORTED_FLAMBY: dict[str, dict[str, Any]] = {
    "HEART": {
        "module": "flamby.datasets.fed_heart_disease",
        "class": "FedHeartDisease",
        "max_clients": 4,
        "num_classes": 2,
        "license": "https://archive.ics.uci.edu/dataset/45/heart+disease",
    },
    "ISIC2019": {
        "module": "flamby.datasets.fed_isic2019",
        "class": "FedIsic2019",
        "max_clients": 6,
        "num_classes": 8,
        "license": "https://challenge.isic-archive.com/data/",
    },
    "IXITINY": {
        "module": "flamby.datasets.fed_ixi",
        "class": "FedIXITiny",
        "max_clients": 3,
        "num_classes": 2,
        "license": "https://brain-development.org/ixi-dataset/",
    },
}


logger = logging.getLogger(__name__)


def _canonical_flamby_key(dataset_name: str) -> str:
    key = str(dataset_name).strip().upper().replace("-", "").replace("_", "")
    aliases = {
        "HEART": "HEART",
        "HEARTDISEASE": "HEART",
        "ISIC2019": "ISIC2019",
        "IXITINY": "IXITINY",
    }
    return aliases.get(key, "")


def _instantiate_flamby_dataset(
    ds_class, *, train: bool, center: int | None, pooled: bool
):
    sig = inspect.signature(ds_class.__init__)
    kwargs = {}
    if "train" in sig.parameters:
        kwargs["train"] = bool(train)
    if "center" in sig.parameters and center is not None:
        kwargs["center"] = int(center)
    if "pooled" in sig.parameters:
        kwargs["pooled"] = bool(pooled)
    return ds_class(**kwargs)


def fetch_flamby(args):
    """FLamby parser with AAggFF-aligned dataset scope.

    Supported datasets: HEART, ISIC2019, IXITINY.
    Others are rejected because additional data-provider approval is required.
    """
    args = to_namespace(args)
    active_logger = resolve_dataset_logger(args, logger)
    split_type = str(getattr(args, "split_type", "pre")).strip().lower()
    if split_type != "pre":
        raise ValueError(
            "For dataset.backend=flamby, split.type must be exactly `pre`."
        )
    key = _canonical_flamby_key(str(args.dataset_name))
    if not key:
        allowed = ", ".join(sorted(_SUPPORTED_FLAMBY.keys()))
        raise PermissionError(
            "Unsupported FLamby dataset for this simulation package. "
            f"Allowed (AAggFF-aligned): {allowed}."
        )

    cfg = _SUPPORTED_FLAMBY[key]
    tag = make_load_tag(key, benchmark="FLAMBY")
    active_logger.info("[%s] resolving dataset module.", tag)
    accepted = bool(getattr(args, "flamby_data_terms_accepted", True))
    if not accepted:
        raise PermissionError(
            f"FLamby dataset '{key}' requires explicit data-term approval. "
            f"Set dataset.configs.terms_accepted=true after accepting terms at: {cfg['license']}"
        )

    try:
        module = __import__(cfg["module"], fromlist=[cfg["class"]])
        ds_class = getattr(module, cfg["class"])
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "flamby is not installed or dataset dependencies are missing. "
            "Install required dependencies for FLamby dataset usage."
        ) from e

    max_clients = int(cfg["max_clients"])
    selected_centers = resolve_fixed_pool_clients(
        available_clients=list(range(max_clients)),
        args=args,
        prefix="flamby",
    )
    if not selected_centers:
        raise ValueError(
            f"No FLamby clients selected for dataset '{key}'. "
            "Check num_clients/subsampling settings."
        )
    active_logger.info("[%s] selected %d centers.", tag, len(selected_centers))

    client_datasets: list[tuple[Any, Any]] = []
    for cid, center_id in enumerate(selected_centers):
        train_ds = _instantiate_flamby_dataset(
            ds_class, train=True, center=int(center_id), pooled=False
        )
        test_ds = _instantiate_flamby_dataset(
            ds_class, train=False, center=int(center_id), pooled=False
        )
        client_datasets.append((train_ds, test_ds))

    # Prefer pooled server eval when supported by dataset API.
    try:
        server_dataset = _instantiate_flamby_dataset(
            ds_class,
            train=False,
            center=None,
            pooled=True,
        )
    except Exception:
        server_dataset = client_datasets[0][1] if client_datasets else None

    args.num_clients = len(client_datasets)
    args.flamby_center_ids = [int(c) for c in selected_centers]
    args.num_classes = int(cfg["num_classes"])
    args.need_embedding = False
    args.seq_len = None
    args.num_embeddings = None

    # Infer tensor shape from first client sample without forcing expensive scans.
    if client_datasets and len(client_datasets[0][0]) > 0:
        sample_x, _ = client_datasets[0][0][0]
        shape = tuple(getattr(sample_x, "shape", ()))
        args.input_shape = shape if shape else (1,)
        if len(args.input_shape) > 1:
            args.in_channels = int(args.input_shape[0])
        else:
            args.in_channels = 1
    else:
        args.input_shape = (1,)
        args.in_channels = 1

    active_logger.info(
        "[%s] finished loading (%d clients).", tag, int(args.num_clients)
    )

    return client_datasets, server_dataset, args
