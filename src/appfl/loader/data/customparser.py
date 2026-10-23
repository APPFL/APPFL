from __future__ import annotations

import importlib
import inspect
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from appfl.loader.data.data_utils import (
    BasicTensorDataset,
    finalize_dataset_outputs,
    make_load_tag,
    package_dataset_outputs,
    resolve_dataset_logger,
    set_common_metadata,
)
from appfl.loader.data.partition_utils import partition_raw_dataset

# TODO: Revisit and finalize the custom dataset parser contract.

logger = logging.getLogger(__name__)


def _to_basic_tensor_dataset(
    payload: Any, name: str, config: Any | None = None
) -> Dataset:
    if isinstance(payload, Dataset):
        return payload

    x_data = None
    y_data = None

    if isinstance(payload, dict):
        for key in ["x", "inputs", "features", "data"]:
            if key in payload:
                x_data = payload[key]
                break
        for key in ["y", "targets", "labels", "label"]:
            if key in payload:
                y_data = payload[key]
                break
    elif isinstance(payload, (tuple, list)) and len(payload) == 2:
        x_data, y_data = payload[0], payload[1]

    if x_data is None or y_data is None:
        raise ValueError(
            "Unable to convert payload into a Dataset. Provide Dataset object or (x, y) tensors/arrays."
        )

    x_tensor = torch.as_tensor(np.asarray(x_data))
    y_tensor = torch.as_tensor(np.asarray(y_data)).long().reshape(-1)

    if x_tensor.shape[0] != y_tensor.shape[0]:
        raise ValueError(
            f"Custom dataset payload shape mismatch: x has {x_tensor.shape[0]} rows, y has {y_tensor.shape[0]} rows"
        )

    if x_tensor.dtype in {torch.int16, torch.int32, torch.int64, torch.uint8}:
        x_tensor = x_tensor.long()
    else:
        x_tensor = x_tensor.float()

    ds = BasicTensorDataset(x_tensor, y_tensor, name=name)
    if isinstance(payload, dict) and config is not None:
        source = str(getattr(config, "pre_source", "")).strip()
        if source != "" and source in payload:
            source_values = np.asarray(payload[source]).reshape(-1)
            if int(source_values.size) != int(y_tensor.shape[0]):
                raise ValueError(
                    f"Custom pre split source '{source}' length mismatch: "
                    f"{int(source_values.size)} vs {int(y_tensor.shape[0])}."
                )
            setattr(ds, source, source_values.astype(object, copy=False))
    return ds


def _normalize_loader_result(result: Any, config: Any):
    if isinstance(result, tuple) and len(result) == 3:
        client_datasets, server_dataset, dataset_meta = result
        dataset_meta = set_common_metadata(dataset_meta, client_datasets)
        return package_dataset_outputs(
            client_datasets=client_datasets,
            server_dataset=server_dataset,
            dataset_meta=dataset_meta,
        )

    if isinstance(result, tuple) and len(result) == 2:
        client_datasets, dataset_meta = result
        dataset_meta = set_common_metadata(dataset_meta, client_datasets)
        return package_dataset_outputs(
            client_datasets=client_datasets,
            server_dataset=None,
            dataset_meta=dataset_meta,
        )

    if isinstance(result, dict):
        if "client_datasets" in result:
            dataset_meta = set_common_metadata(
                result.get("dataset_meta", config),
                result["client_datasets"],
            )
            return package_dataset_outputs(
                client_datasets=result["client_datasets"],
                server_dataset=result.get("server_dataset", None),
                dataset_meta=dataset_meta,
            )

    raise ValueError(
        "Custom parser expects (client_datasets, server_dataset, dataset_meta), "
        "(client_datasets, dataset_meta), or dict with client_datasets "
        "(optional: server_dataset, dataset_meta)."
    )


def _load_from_callable(config):
    loader_spec = str(getattr(config, "custom_entrypoint", "")).strip()
    if ":" not in loader_spec:
        raise ValueError(
            "dataset.configs.entrypoint must be in 'package.module:function' format."
        )

    module_name, fn_name = loader_spec.split(":", 1)
    fn = getattr(importlib.import_module(module_name), fn_name)

    kwargs_raw = getattr(config, "custom_kwargs", {})
    if isinstance(kwargs_raw, dict):
        kwargs = dict(kwargs_raw)
    elif isinstance(kwargs_raw, str):
        kwargs = json.loads(kwargs_raw) if kwargs_raw.strip() else {}
    else:
        kwargs = {}

    sig = inspect.signature(fn)
    if "config" in sig.parameters:
        result = fn(config=config, **kwargs)
    elif "cfg" in sig.parameters:
        result = fn(cfg=vars(config), **kwargs)
    else:
        result = fn(**kwargs)

    return _normalize_loader_result(result, config)


def _load_train_test_from_directory(
    data_dir: Path, config: Any
) -> tuple[Dataset, Dataset | None]:
    train_candidates = [
        data_dir / "train.pt",
        data_dir / "train.pth",
        data_dir / "train.npz",
    ]
    test_candidates = [
        data_dir / "test.pt",
        data_dir / "test.pth",
        data_dir / "test.npz",
    ]

    train_obj = None
    for path in train_candidates:
        if path.exists():
            if path.suffix == ".npz":
                with np.load(path) as npz:
                    train_obj = {"x": npz["x"], "y": npz["y"]}
            else:
                train_obj = torch.load(path, map_location="cpu")
            break

    if train_obj is None:
        raise FileNotFoundError(
            f"No train artifact found under {data_dir}. Expected one of: {[p.name for p in train_candidates]}"
        )

    test_obj = None
    for path in test_candidates:
        if path.exists():
            if path.suffix == ".npz":
                with np.load(path) as npz:
                    test_obj = {"x": npz["x"], "y": npz["y"]}
            else:
                test_obj = torch.load(path, map_location="cpu")
            break

    train_ds = _to_basic_tensor_dataset(train_obj, "[CUSTOM] TRAIN", config=config)
    test_ds = (
        _to_basic_tensor_dataset(test_obj, "[CUSTOM] TEST", config=config)
        if test_obj is not None
        else None
    )
    return train_ds, test_ds


def _load_from_path(config):
    custom_path = Path(str(getattr(config, "data_dir", "")).strip()).expanduser()
    if not custom_path.exists():
        raise FileNotFoundError(f"dataset.path does not exist: {custom_path}")

    if custom_path.is_file():
        if custom_path.suffix == ".npz":
            with np.load(custom_path) as npz:
                payload = {"x": npz["x"], "y": npz["y"]}
        else:
            payload = torch.load(custom_path, map_location="cpu")
        try:
            return _normalize_loader_result(payload, config)
        except (TypeError, ValueError, KeyError):
            train_ds = _to_basic_tensor_dataset(
                payload, "[CUSTOM] TRAIN", config=config
            )
            client_datasets = partition_raw_dataset(train_ds, config)
            return finalize_dataset_outputs(
                client_datasets=client_datasets,
                server_dataset=None,
                dataset_meta=config,
                raw_train=train_ds,
            )

    candidate = custom_path / "dataset.pt"
    if candidate.exists():
        payload = torch.load(candidate, map_location="cpu")
        try:
            return _normalize_loader_result(payload, config)
        except (TypeError, ValueError, KeyError):
            pass

    client_payload = custom_path / "client_datasets.pt"
    if client_payload.exists():
        payload = torch.load(client_payload, map_location="cpu")
        if isinstance(payload, dict) and "client_datasets" in payload:
            return _normalize_loader_result(payload, config)
        if isinstance(payload, list):
            dataset_meta = set_common_metadata(config, payload)
            return package_dataset_outputs(
                client_datasets=payload,
                server_dataset=None,
                dataset_meta=dataset_meta,
            )

    train_ds, test_ds = _load_train_test_from_directory(custom_path, config=config)
    client_datasets = partition_raw_dataset(train_ds, config)
    return finalize_dataset_outputs(
        client_datasets=client_datasets,
        server_dataset=test_ds,
        dataset_meta=config,
        raw_train=train_ds,
    )


def fetch_custom_dataset(config):
    """Custom dataset parser.

    Supports two modes:
    - `dataset.configs.entrypoint=package.module:function` callable contract.
    - `dataset.path=/path/to/data_or_artifact` local artifact contract.
    """
    active_logger = resolve_dataset_logger(config, logger)
    tag = make_load_tag(
        str(getattr(config, "dataset_name", "custom")), benchmark="CUSTOM"
    )

    loader_spec = str(getattr(config, "custom_entrypoint", "")).strip()
    dataset_path = str(getattr(config, "data_dir", "")).strip()

    if loader_spec:
        active_logger.info("[%s] loading via custom callable.", tag)
        out = _load_from_callable(config)
        active_logger.info("[%s] finished loading.", tag)
        return out
    if dataset_path:
        active_logger.info("[%s] loading from custom path.", tag)
        out = _load_from_path(config)
        active_logger.info("[%s] finished loading.", tag)
        return out

    raise ValueError(
        "For dataset.backend=custom, set either dataset.configs.entrypoint or dataset.path."
    )
