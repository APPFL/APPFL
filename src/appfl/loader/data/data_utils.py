from __future__ import annotations
import ast
import importlib
import logging
import random
import traceback
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, Dataset, Subset, random_split
from appfl.logger import ServerAgentFileLogger


class _DatasetLoggerAdapter:
    """Adapter for APPFL custom loggers to stdlib-like interface."""

    def __init__(self, target):
        self._target = target

    @staticmethod
    def _fmt(msg: str, *args) -> str:
        if args:
            try:
                return str(msg) % args
            except Exception:
                return f"{msg} {' '.join(str(a) for a in args)}"
        return str(msg)

    def info(self, msg, *args, **kwargs):
        self._target.info(self._fmt(msg, *args))

    def debug(self, msg, *args, **kwargs):
        if hasattr(self._target, "debug"):
            self._target.debug(self._fmt(msg, *args))
        else:
            self._target.info(self._fmt(msg, *args))

    def warning(self, msg, *args, **kwargs):
        if hasattr(self._target, "warning"):
            self._target.warning(self._fmt(msg, *args))
        else:
            self._target.info(self._fmt(msg, *args))

    def error(self, msg, *args, **kwargs):
        if hasattr(self._target, "error"):
            self._target.error(self._fmt(msg, *args))
        else:
            self._target.info(self._fmt(msg, *args))

    def exception(self, msg, *args, **kwargs):
        text = self._fmt(msg, *args)
        exc_text = traceback.format_exc()
        if exc_text and exc_text.strip() != "NoneType: None":
            text = f"{text}\n{exc_text}"
        if hasattr(self._target, "error"):
            self._target.error(text)
        else:
            self._target.info(text)


def resolve_dataset_logger(config: Any, default_logger: logging.Logger):
    candidate = getattr(config, "logger", None)
    if candidate is None:
        return default_logger
    if isinstance(candidate, logging.Logger):
        return candidate
    return _DatasetLoggerAdapter(candidate)


def make_load_tag(dataset_name: str, benchmark: str | None = None) -> str:
    ds = str(dataset_name).strip().upper()
    bench = str(benchmark or "").strip().upper()
    return f"{bench}-{ds}" if bench else ds


class BasicTensorDataset(Dataset):
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor, name: str = "dataset"):
        self.inputs = inputs
        self.targets = _coerce_target_tensor(targets)
        self.name = name

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int):
        return self.inputs[index], self.targets[index]

    def __repr__(self) -> str:
        return self.name


def _coerce_target_tensor(targets: Any) -> torch.Tensor:
    if torch.is_tensor(targets):
        tensor = targets.detach().clone()
    else:
        tensor = torch.as_tensor(targets)
    return tensor.float() if tensor.dtype.is_floating_point else tensor.long()


def infer_input_shape(dataset: Dataset) -> Tuple[int, ...]:
    x, _ = dataset[0]
    if not torch.is_tensor(x):
        x = torch.as_tensor(np.asarray(x))
    return tuple(x.shape)


def extract_targets(dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        targets = dataset.targets
        if torch.is_tensor(targets):
            return targets.detach().cpu().numpy()
        return np.asarray(targets)

    if hasattr(dataset, "tensors") and len(dataset.tensors) >= 2:
        return dataset.tensors[1].detach().cpu().numpy()

    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        parent_targets = extract_targets(dataset.dataset)
        return parent_targets[np.asarray(dataset.indices)]

    labels = []
    for i in range(len(dataset)):
        sample = dataset[i]
        if not isinstance(sample, (tuple, list)) or len(sample) < 2:
            raise ValueError("Unable to extract targets from dataset.")
        y = sample[1]
        if torch.is_tensor(y):
            if y.ndim == 0:
                labels.append(int(y.item()))
            else:
                labels.append(int(y.reshape(-1)[0].item()))
        elif isinstance(y, np.ndarray):
            labels.append(int(np.asarray(y).reshape(-1)[0]))
        else:
            labels.append(int(y))
    return np.asarray(labels, dtype=np.int64)


def infer_num_classes(dataset) -> int:
    targets = extract_targets(dataset)
    return int(np.unique(targets).size) if targets.size > 0 else 0


def concat_targets(datasets: Iterable[Dataset]) -> np.ndarray:
    chunks = []
    for ds in datasets:
        if ds is None or len(ds) == 0:
            continue
        chunks.append(extract_targets(ds))
    if not chunks:
        return np.asarray([], dtype=np.int64)
    return np.concatenate(chunks).astype(np.int64)


def _train_dataset_from_entry(entry):
    if not isinstance(entry, tuple) or len(entry) < 1:
        return None
    return entry[0]


def _first_nonempty_train_dataset(client_datasets):
    for entry in client_datasets:
        train_ds = _train_dataset_from_entry(entry)
        if train_ds is not None and len(train_ds) > 0:
            return train_ds
    return None


def package_dataset_outputs(
    client_datasets,
    server_dataset,
    dataset_meta: Any,
):
    return client_datasets, server_dataset, dataset_meta


def finalize_dataset_outputs(
    client_datasets,
    server_dataset,
    dataset_meta: Any,
    raw_train: Dataset | None = None,
):
    dataset_meta = set_common_metadata(
        dataset_meta,
        client_datasets,
        raw_train=raw_train,
    )
    return package_dataset_outputs(
        client_datasets=client_datasets,
        server_dataset=server_dataset,
        dataset_meta=dataset_meta,
    )


def set_common_metadata(
    config: Any,
    client_datasets,
    raw_train: Dataset | None = None,
):
    config.num_clients = len(client_datasets)

    shape_source = raw_train if raw_train is not None and len(raw_train) > 0 else _first_nonempty_train_dataset(client_datasets)
    config.input_shape = infer_input_shape(shape_source) if shape_source is not None else (1,)

    if raw_train is not None and len(raw_train) > 0:
        config.num_classes = infer_num_classes(raw_train)
    else:
        train_targets = concat_targets(
            train_ds
            for train_ds in (
                _train_dataset_from_entry(entry) for entry in client_datasets
            )
            if train_ds is not None
        )
        config.num_classes = int(np.unique(train_targets).size) if train_targets.size > 0 else 0

    if getattr(config, "in_channels", None) is None:
        if len(config.input_shape) == 1:
            config.in_channels = 1
        elif len(config.input_shape) > 1:
            config.in_channels = int(config.input_shape[0])
    return config


def _cfg_get(config, path: str, default=None):
    current = config
    for part in path.split("."):
        if isinstance(current, dict):
            if part not in current:
                return default
            current = current[part]
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            return default
    return default if current is None else current


def _cfg_set(config, path: str, value) -> None:
    parts = path.split(".")
    current = config
    for part in parts[:-1]:
        if isinstance(current, dict):
            current = current.setdefault(part, {})
        else:
            if not hasattr(current, part):
                setattr(current, part, {})
            current = getattr(current, part)
    if isinstance(current, dict):
        current[parts[-1]] = value
    else:
        setattr(current, parts[-1], value)


def _safe_int(value: Any, default: int) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _safe_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
        return bool(default)
    return bool(value)


def resolve_fixed_pool_clients(
    available_clients: List[Any],
    config,
    prefix: str = "",
) -> List[Any]:
    """Resolve client subset for fixed-pool datasets such as LEAF."""
    del prefix
    pool = list(available_clients)
    if not pool:
        return []

    infer_clients = _safe_bool(getattr(config, "pre_infer_num_clients", False), False)
    requested_num = _safe_int(getattr(config, "num_clients", 0), 0)
    if infer_clients or requested_num <= 0:
        requested_num = len(pool)
    requested_num = max(1, min(requested_num, len(pool)))
    return pool[:requested_num]


def _resolve_algorithm_components(config) -> Dict[str, str]:
    return {
        "scheduler_name": str(
            _cfg_get(
                config,
                "algorithm_configs.scheduler",
                _cfg_get(config, "server_configs.scheduler", ""),
            )
        )
    }


def _parse_holdout_dataset_ratio(config: DictConfig) -> Optional[List[float]]:
    raw = _cfg_get(config, "eval.configs.dataset_ratio", None)
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if text == "":
            return None
        try:
            parsed = ast.literal_eval(text)
        except Exception as exc:
            raise ValueError(
                "eval.configs.dataset_ratio must be a list-like string, e.g. '[80,20]' or '[0.8,0.1,0.1]'"
            ) from exc
    else:
        parsed = raw

    if isinstance(parsed, (int, float)):
        raise ValueError("eval.configs.dataset_ratio must contain 1, 2, or 3 values.")
    ratios = [float(x) for x in parsed]
    if len(ratios) not in {1, 2, 3}:
        raise ValueError("eval.configs.dataset_ratio must have length 1, 2, or 3.")
    if any(x <= 0 for x in ratios):
        raise ValueError("eval.configs.dataset_ratio values must be positive.")
    total = float(sum(ratios))
    if np.isclose(total, 100.0, atol=1e-6):
        ratios = [x / 100.0 for x in ratios]
    elif not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(
            "eval.configs.dataset_ratio must sum to 1.0 or 100.0, e.g. [0.8,0.2] or [80,20]."
        )
    if len(ratios) == 1 and not np.isclose(ratios[0], 1.0, atol=1e-6):
        raise ValueError(
            "eval.configs.dataset_ratio with a single value only accepts [1.0] or [100]."
        )
    return ratios

def _safe_split_lengths(n: int, ratios: List[float]) -> List[int]:
    lengths = [int(float(n) * r) for r in ratios]
    remain = int(n) - int(sum(lengths))
    for i in range(remain):
        lengths[i % len(lengths)] += 1
    # If possible, ensure each partition has at least one sample.
    if n >= len(lengths):
        for idx in range(len(lengths)):
            if lengths[idx] > 0:
                continue
            donor = int(np.argmax(lengths))
            if lengths[donor] > 1:
                lengths[donor] -= 1
                lengths[idx] = 1
    return lengths


def _splits_have_nonempty_parts(splits, expected_parts: int) -> bool:
    if splits is None or len(splits) < int(expected_parts):
        return False
    for idx in range(int(expected_parts)):
        try:
            if int(len(splits[idx])) <= 0:
                return False
        except Exception:
            return False
    return True


def _dataset_targets(dataset) -> Optional[np.ndarray]:
    def _as_label_array(value) -> Optional[np.ndarray]:
        if value is None:
            return None
        if torch.is_tensor(value):
            arr = value.detach().cpu().numpy()
        else:
            arr = np.asarray(value)
        if arr.size == 0:
            return np.asarray([], dtype=np.int64)
        return arr.reshape(-1).astype(np.int64, copy=False)

    def _subset_indices_array(indices, subset_len: int) -> Optional[np.ndarray]:
        if torch.is_tensor(indices):
            idx = indices.detach().cpu().numpy()
        else:
            idx = np.asarray(indices)
        if idx.size != subset_len:
            return None
        return idx.reshape(-1).astype(np.int64, copy=False)

    try:
        n = int(len(dataset))
    except Exception:
        return None
    if n <= 0:
        return np.asarray([], dtype=np.int64)

    for attr_name in ("targets", "labels", "y"):
        arr = _as_label_array(getattr(dataset, attr_name, None))
        if arr is not None and arr.size == n:
            return arr

    if isinstance(dataset, Subset):
        base_labels = _dataset_targets(dataset.dataset)
        if base_labels is not None:
            idx = _subset_indices_array(dataset.indices, n)
            if idx is not None:
                try:
                    return base_labels[idx]
                except Exception:
                    pass

    if isinstance(dataset, ConcatDataset):
        parts: List[np.ndarray] = []
        total = 0
        for child in dataset.datasets:
            child_labels = _dataset_targets(child)
            if child_labels is None:
                parts = []
                break
            parts.append(child_labels)
            total += int(child_labels.size)
        if parts and total == n:
            return np.concatenate(parts).astype(np.int64, copy=False)

    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        try:
            item = dataset[i]
        except Exception:
            return None
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            return None
        y = item[1]
        if torch.is_tensor(y):
            if y.numel() == 0:
                return None
            y = y.reshape(-1)[0].item()
        elif isinstance(y, np.ndarray):
            if y.size == 0:
                return None
            y = y.reshape(-1)[0].item()
        out[i] = int(y)
    return out


def _stratified_split_dataset(
    dataset,
    ratios: List[float],
    seed: int,
):
    total = int(len(dataset))
    labels = _dataset_targets(dataset)
    if labels is None or labels.size != total:
        return None

    rng = np.random.default_rng(seed)
    num_parts = len(ratios)
    split_bins = [[] for _ in range(num_parts)]
    all_indices = np.arange(total, dtype=np.int64)
    classes = np.unique(labels)

    for cls in classes:
        cls_idx = all_indices[labels == cls]
        if cls_idx.size == 0:
            continue
        cls_idx = rng.permutation(cls_idx)
        expected = np.asarray(ratios, dtype=float) * float(cls_idx.size)
        counts = np.floor(expected).astype(int)
        remain = int(cls_idx.size - counts.sum())
        if remain > 0:
            order = np.argsort(-(expected - counts))
            for j in range(remain):
                counts[int(order[j % num_parts])] += 1
        cursor = 0
        for part_id, cnt in enumerate(counts.tolist()):
            if cnt <= 0:
                continue
            split_bins[part_id].append(cls_idx[cursor : cursor + cnt])
            cursor += cnt

    subsets = []
    for part in split_bins:
        if part:
            idx = np.concatenate(part).astype(np.int64, copy=False)
            idx = rng.permutation(idx)
        else:
            idx = np.asarray([], dtype=np.int64)
        subsets.append(Subset(dataset, idx.tolist()))
    return subsets

def _normalize_client_tuple(entry) -> Tuple[Optional[object], Optional[object], Optional[object]]:
    if not isinstance(entry, tuple):
        raise ValueError("Each client dataset entry must be a tuple.")
    if len(entry) == 1:
        train_ds = entry[0]
        return train_ds, None, None
    if len(entry) == 2:
        train_ds, test_ds = entry
        return train_ds, None, test_ds
    if len(entry) == 3:
        train_ds, val_ds, test_ds = entry
        return train_ds, val_ds, test_ds
    raise ValueError(
        "Each client dataset entry must be tuple(train), tuple(train,test), or tuple(train,val,test)."
    )

def _apply_holdout_dataset_ratio(
    client_datasets,
    config: DictConfig,
    logger: Optional[ServerAgentFileLogger] = None,
):
    ratios = _parse_holdout_dataset_ratio(config)
    if ratios is None:
        return client_datasets
    train_only = len(ratios) == 1
    if train_only:
        _cfg_set(config, "eval.do_pre_evaluation", False)
        _cfg_set(config, "eval.do_post_evaluation", False)
        _cfg_set(config, "eval.enable_global_eval", False)
        _cfg_set(config, "eval.enable_federated_eval", False)
        if logger is not None:
            logger.info(
                "eval.configs.dataset_ratio=[1.0|100] detected: disabling validation/test "
                "and global/federated evaluation (training metrics only)."
            )

    seed = int(_cfg_get(config, "data_configs.partition_kwargs.data_seed", 0))
    out = []
    for cid, entry in enumerate(client_datasets):
        train_ds, val_ds, test_ds = _normalize_client_tuple(entry)
        parts = [ds for ds in (train_ds, val_ds, test_ds) if ds is not None]
        if not parts:
            raise ValueError(f"Client dataset entry {cid} is empty.")
        merged = parts[0] if len(parts) == 1 else ConcatDataset(parts)
        total = len(merged)
        if total <= 0:
            if train_only:
                out.append((merged, None, None))
            elif len(ratios) == 2:
                out.append((merged, merged))
            else:
                out.append((merged, merged, merged))
            continue
        split_seed = seed + 7919 + int(cid)
        splits = _stratified_split_dataset(
            merged,
            ratios=ratios,
            seed=split_seed,
        )
        if splits is None or (
            total >= len(ratios) and not _splits_have_nonempty_parts(splits, len(ratios))
        ):
            lengths = _safe_split_lengths(total, ratios)
            generator = torch.Generator().manual_seed(split_seed)
            splits = random_split(merged, lengths, generator=generator)
        if train_only:
            out.append((splits[0], None, None))
        elif len(ratios) == 2:
            out.append((splits[0], splits[1]))
        else:
            out.append((splits[0], splits[1], splits[2]))
    del logger
    return out

def _dataset_has_eval_split(dataset) -> bool:
    """Return whether a dataset is present and plausibly non-empty for evaluation."""
    if dataset is None:
        return False
    # Most map-style datasets implement __len__; trust it when available.
    try:
        return int(len(dataset)) > 0
    except Exception:
        # Iterable/streaming datasets may not expose length; if object exists, allow eval path.
        return True

def _validate_loader_output(client_datasets, runtime_cfg: Dict) -> None:
    num_clients = int(runtime_cfg["num_clients"])
    if len(client_datasets) != num_clients:
        raise ValueError(
            f"Loader/client metadata mismatch: len(client_datasets)={len(client_datasets)} "
            f"but num_clients={num_clients}"
        )
    for cid, pair in enumerate(client_datasets):
        if not (isinstance(pair, tuple) and len(pair) in {1, 2, 3}):
            raise ValueError(
                f"client_datasets[{cid}] must be tuple(train), tuple(train,test), or tuple(train,val,test)."
            )

def _build_client_groups(config: DictConfig, num_clients: int) -> Tuple[List[int], List[int]]:
    all_clients = list(range(int(num_clients)))
    scheme = str(_cfg_get(config, "eval.configs.scheme", "dataset")).strip().lower()
    if scheme != "client":
        return all_clients, []

    holdout_num = int(_cfg_get(config, "eval.configs.client_counts", 0))
    holdout_ratio = float(_cfg_get(config, "eval.configs.client_ratio", 0.0))
    if holdout_num <= 0 and holdout_ratio > 0.0:
        holdout_num = max(1, int(round(num_clients * holdout_ratio)))
    holdout_num = max(0, min(holdout_num, max(0, num_clients - 1)))
    if holdout_num == 0:
        return all_clients, []

    rng = random.Random(
        int(_cfg_get(config, "data_configs.partition_kwargs.data_seed", 42)) + 2026
    )
    shuffled = all_clients[:]
    rng.shuffle(shuffled)
    holdout = sorted(shuffled[:holdout_num])
    train_clients = sorted(cid for cid in all_clients if cid not in set(holdout))
    if not train_clients:
        return all_clients, []
    return train_clients, holdout

def _sample_train_clients(train_client_ids: List[int], num_sampled_clients: int) -> List[int]:
    if not train_client_ids:
        return []
    n = max(1, int(num_sampled_clients))
    n = min(n, len(train_client_ids))
    return sorted(random.sample(train_client_ids, n))

def _resolve_num_sampled_clients(config: DictConfig, num_clients: int) -> int:
    if int(num_clients) <= 0:
        return 0

    raw = _cfg_get(config, "train.num_sampled_clients", None)
    if raw is not None:
        try:
            n = int(raw)
        except Exception:
            n = 0
        if n > 0:
            return max(1, min(int(num_clients), n))

    return int(num_clients)

def _resolve_client_eval_dataset(
    client_datasets: Sequence,
    client_id: int,
    eval_split: str,
):
    train_ds, val_ds, test_ds = _normalize_client_tuple(client_datasets[int(client_id)])
    del train_ds
    chosen = str(eval_split).strip().lower()
    if chosen in {"val", "validation"}:
        return val_ds if val_ds is not None else test_ds
    return test_ds if test_ds is not None else val_ds

def _validate_required_dataset_ratio(config: DictConfig) -> None:
    ratios = _parse_holdout_dataset_ratio(config)
    if ratios is None:
        raise ValueError(
            "This scheduler requires `eval.configs.dataset_ratio` including "
            "a validation split, e.g. [80,10,10]."
        )
    if len(ratios) < 3:
        raise ValueError(
            "This scheduler requires `eval.configs.dataset_ratio` with "
            "three entries (train/val/test), e.g. [80,10,10]."
        )


def _resolve_scheduler_required_data_fields(config: DictConfig) -> set[str]:
    components = _resolve_algorithm_components(config)
    scheduler_name = str(components["scheduler_name"]).strip()
    if scheduler_name == "":
        return set()
    module = importlib.import_module("appfl.algorithm.scheduler")
    scheduler_cls = getattr(module, scheduler_name)
    fields = (
        scheduler_cls.required_data_fields()
        if hasattr(scheduler_cls, "required_data_fields")
        else set()
    )
    return {str(field).strip() for field in fields if str(field).strip()}


def _validate_algorithm_data_requirements(config: DictConfig) -> None:
    validators = {
        "eval.configs.dataset_ratio": _validate_required_dataset_ratio,
    }
    for field in sorted(_resolve_scheduler_required_data_fields(config)):
        if field not in validators:
            raise ValueError(
                f"Unsupported scheduler data requirement `{field}`. "
                "Add a corresponding validator in appfl.loader.data.data_utils."
            )
        validators[field](config)
