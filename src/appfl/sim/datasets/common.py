from __future__ import annotations
from dataclasses import dataclass, field
import logging
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


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


def resolve_dataset_logger(args: Any, default_logger: logging.Logger):
    candidate = getattr(args, "logger", None)
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
        self.targets = targets.long()
        self.name = name

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int):
        return self.inputs[index], self.targets[index]

    def __repr__(self) -> str:
        return self.name


@dataclass
class DatasetArgs:
    num_clients: int = 20
    seed: int = 42
    dataset_name: str = "MNIST"
    dataset_backend: str = "torchvision"
    data_dir: str = "./data"
    download: bool = True
    test_size: float = 0.2
    split_type: str = "iid"
    dirichlet_alpha: float = 0.3
    min_classes: int = 2
    unbalanced_keep_min: float = 0.5
    pre_infer_num_clients: bool = False
    pre_source: str = ""
    pre_index: int = -1
    seq_len: int = 128
    num_embeddings: int = 10000
    use_model_tokenizer: bool = False
    model_name: str = "SimpleCNN"
    in_channels: int | None = None
    audio_num_frames: int = 16000
    flamby_data_terms_accepted: bool = True
    leaf_raw_data_fraction: float = 1.0
    leaf_min_samples_per_client: int = 2
    leaf_image_root: str = ""
    ext_source: str = ""
    ext_dataset_name: str = ""
    ext_train_split: str = "train"
    ext_test_split: str = "test"
    ext_feature_key: str = ""
    ext_label_key: str = ""
    ext_config_name: str = ""
    custom_entrypoint: str = ""
    custom_kwargs: Dict[str, Any] = field(default_factory=dict)
    logger: Any = None


def _get_path(payload: Any, path: str, default: Any) -> Any:
    parts = [p for p in str(path).split(".") if p]
    cur = payload
    for part in parts:
        if isinstance(cur, dict):
            if part not in cur:
                return default
            cur = cur[part]
            continue
        if hasattr(cur, part):
            cur = getattr(cur, part)
            continue
        return default
    return default if cur is None else cur


def _coerce_num_clients(raw: Any, fallback: int = 0) -> int:
    try:
        if raw is None:
            return int(fallback)
        return int(raw)
    except Exception:
        return int(fallback)


def _as_mapping(args: Any) -> Dict[str, Any]:
    if isinstance(args, dict):
        return dict(args)
    return dict(vars(args))


def to_namespace(args: Any) -> DatasetArgs:
    if isinstance(args, DatasetArgs):
        ns = DatasetArgs()
        ns.__dict__.update(vars(args))
        ns.num_clients = _coerce_num_clients(getattr(ns, "num_clients", None), 20)
        return ns
    if hasattr(args, "dataset_name") and hasattr(args, "data_dir"):
        payload = _as_mapping(args)
        ns = DatasetArgs()
        ns.__dict__.update(payload)
        ns.num_clients = _coerce_num_clients(getattr(ns, "num_clients", None), 20)
        return ns
    if isinstance(args, dict) and "dataset_name" in args and "data_dir" in args:
        ns = DatasetArgs()
        ns.__dict__.update(dict(args))
        ns.num_clients = _coerce_num_clients(getattr(ns, "num_clients", None), 20)
        return ns

    payload: Any = _as_mapping(args)

    dataset_cfg = _get_path(payload, "dataset.configs", {})
    model_cfg = _get_path(payload, "model.configs", {})
    split_cfg = _get_path(payload, "split.configs", {})

    eval_dataset_ratio = _get_path(payload, "eval.configs.dataset_ratio", [80, 20])
    try:
        test_size = float(eval_dataset_ratio[-1]) if len(eval_dataset_ratio) >= 2 else 0.2
        if test_size > 1.0:
            test_size = test_size / 100.0
    except Exception:
        test_size = 0.2

    num_clients = _coerce_num_clients(
        _get_path(payload, "train.num_clients", 20),
        20,
    )

    return DatasetArgs(
        num_clients=num_clients,
        seed=int(_get_path(payload, "experiment.seed", 42)),
        dataset_name=str(_get_path(payload, "dataset.name", "MNIST")),
        dataset_backend=str(_get_path(payload, "dataset.backend", "torchvision")),
        data_dir=str(_get_path(payload, "dataset.path", "./data")),
        download=bool(_get_path(payload, "dataset.download", True)),
        test_size=float(test_size),
        split_type=str(_get_path(payload, "split.type", "iid")),
        dirichlet_alpha=float(_get_path(split_cfg, "dirichlet_alpha", 0.3)),
        min_classes=int(_get_path(split_cfg, "min_classes", 2)),
        unbalanced_keep_min=float(_get_path(split_cfg, "unbalanced_keep_min", 0.5)),
        pre_infer_num_clients=bool(_get_path(split_cfg, "pre_infer_num_clients", False)),
        pre_source=str(_get_path(split_cfg, "pre_source", "")),
        pre_index=int(_get_path(split_cfg, "pre_index", -1)),
        seq_len=int(_get_path(model_cfg, "seq_len", 128)),
        num_embeddings=int(_get_path(model_cfg, "num_embeddings", 10000)),
        use_model_tokenizer=bool(_get_path(model_cfg, "use_model_tokenizer", False)),
        model_name=str(_get_path(payload, "model.name", "SimpleCNN")),
        in_channels=_get_path(model_cfg, "in_channels", None),
        audio_num_frames=int(_get_path(dataset_cfg, "audio_num_frames", 16000)),
        flamby_data_terms_accepted=bool(_get_path(dataset_cfg, "terms_accepted", True)),
        leaf_raw_data_fraction=float(_get_path(dataset_cfg, "raw_data_fraction", 1.0)),
        leaf_min_samples_per_client=int(_get_path(dataset_cfg, "min_samples_per_client", 2)),
        leaf_image_root=str(_get_path(dataset_cfg, "image_root", "")),
        ext_source=str(_get_path(dataset_cfg, "source", "")),
        ext_dataset_name=str(_get_path(dataset_cfg, "dataset_name", "")),
        ext_train_split=str(_get_path(dataset_cfg, "train_split", "train")),
        ext_test_split=str(_get_path(dataset_cfg, "test_split", "test")),
        ext_feature_key=str(_get_path(dataset_cfg, "feature_key", "")),
        ext_label_key=str(_get_path(dataset_cfg, "label_key", "")),
        ext_config_name=str(_get_path(dataset_cfg, "config_name", "")),
        custom_entrypoint=str(_get_path(dataset_cfg, "entrypoint", "")),
        custom_kwargs=_get_path(dataset_cfg, "kwargs", {}),
    )


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
    args: DatasetArgs,
    prefix: str = "",
) -> List[Any]:
    """Resolve client subset for fixed-pool datasets (LEAF/FLamby/TFF)."""
    del prefix
    pool = list(available_clients)
    if not pool:
        return []

    infer_clients = _safe_bool(getattr(args, "pre_infer_num_clients", False), False)
    requested_num = _safe_int(getattr(args, "num_clients", 0), 0)
    if infer_clients or requested_num <= 0:
        requested_num = len(pool)
    requested_num = max(1, min(requested_num, len(pool)))
    return pool[:requested_num]


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
    dataset_meta: DatasetArgs,
):
    return client_datasets, server_dataset, dataset_meta


def finalize_dataset_outputs(
    client_datasets,
    server_dataset,
    dataset_meta: Any,
    raw_train: Dataset | None = None,
):
    meta_ns = to_namespace(dataset_meta)
    meta_ns = set_common_metadata(
        meta_ns,
        client_datasets,
        raw_train=raw_train,
    )
    return package_dataset_outputs(
        client_datasets=client_datasets,
        server_dataset=server_dataset,
        dataset_meta=meta_ns,
    )


def _iid_split(num_samples: int, num_clients: int, rng: np.random.Generator) -> dict[int, np.ndarray]:
    perm = rng.permutation(num_samples)
    chunks = np.array_split(perm, num_clients)
    return {cid: chunk.astype(np.int64) for cid, chunk in enumerate(chunks)}


def _unbalanced_split(
    num_samples: int,
    num_clients: int,
    rng: np.random.Generator,
    keep_min: float,
) -> dict[int, np.ndarray]:
    base = _iid_split(num_samples, num_clients, rng)
    out: dict[int, np.ndarray] = {}
    for cid, indices in base.items():
        if len(indices) <= 1:
            out[cid] = indices
            continue
        keep_ratio = rng.uniform(keep_min, 1.0)
        keep_count = max(1, int(len(indices) * keep_ratio))
        out[cid] = indices[:keep_count]
    return out


def _pathological_split(
    labels: np.ndarray,
    num_clients: int,
    min_classes: int,
    rng: np.random.Generator,
) -> dict[int, np.ndarray]:
    cap = max(1, int(min_classes))
    classes = np.asarray(sorted(np.unique(labels)), dtype=np.int64)
    if classes.size == 0:
        return {cid: np.array([], dtype=np.int64) for cid in range(num_clients)}

    total_slots = int(num_clients) * int(cap)
    if total_slots < int(classes.size):
        raise ValueError(
            "pathological split cannot cover all classes with current settings: "
            f"num_clients({num_clients}) * min_classes({cap}) < num_classes({int(classes.size)})."
        )

    # Each client gets `cap` class slots; every class is assigned to at least one slot.
    client_slots = rng.permutation(np.repeat(np.arange(num_clients, dtype=np.int64), cap))
    class_to_clients: dict[int, list[int]] = {int(cls): [] for cls in classes.tolist()}
    client_class_sets = [set() for _ in range(num_clients)]

    shuffled_classes = rng.permutation(classes)
    for cls, cid in zip(shuffled_classes.tolist(), client_slots[: classes.size].tolist()):
        c = int(cid)
        y = int(cls)
        client_class_sets[c].add(y)
        class_to_clients[y].append(c)

    for cid in client_slots[classes.size :].tolist():
        c = int(cid)
        available = [int(cls) for cls in classes.tolist() if int(cls) not in client_class_sets[c]]
        if not available:
            continue
        y = int(rng.choice(np.asarray(available, dtype=np.int64)))
        client_class_sets[c].add(y)
        class_to_clients[y].append(c)

    out = {cid: [] for cid in range(num_clients)}
    for cls in classes.tolist():
        y = int(cls)
        cls_idx = np.where(labels == y)[0].astype(np.int64)
        rng.shuffle(cls_idx)
        owners = class_to_clients[y]
        if not owners:
            raise RuntimeError(f"pathological split internal error: class {y} has no owner.")
        parts = np.array_split(cls_idx, len(owners))
        for cid, part in zip(owners, parts):
            if len(part) > 0:
                out[int(cid)].append(part.astype(np.int64))

    result = {
        cid: np.concatenate(parts).astype(np.int64) if parts else np.array([], dtype=np.int64)
        for cid, parts in out.items()
    }

    # Verify constraints: per-client class cap and global class coverage.
    covered = set()
    for cid, idx in result.items():
        if idx.size == 0:
            continue
        uniq = set(np.unique(labels[idx]).astype(np.int64).tolist())
        if len(uniq) > cap:
            raise RuntimeError(
                f"pathological split produced {len(uniq)} classes for client {cid}, cap={cap}."
            )
        covered.update(int(v) for v in uniq)
    expected = set(int(v) for v in classes.tolist())
    if covered != expected:
        raise RuntimeError("pathological split failed to cover all classes across clients.")

    return result


def _dirichlet_split(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    min_size: int,
    rng: np.random.Generator,
    max_retry: int = 20,
) -> dict[int, np.ndarray]:
    classes = np.unique(labels)
    alpha = max(alpha, 1e-3)

    for _ in range(max_retry):
        splits = [[] for _ in range(num_clients)]
        for cls in classes:
            cls_idx = np.where(labels == cls)[0]
            rng.shuffle(cls_idx)
            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            cut_points = (np.cumsum(proportions)[:-1] * len(cls_idx)).astype(int)
            cls_splits = np.split(cls_idx, cut_points)
            for cid, part in enumerate(cls_splits):
                splits[cid].append(part)

        result = {
            cid: np.concatenate(parts).astype(np.int64) if parts else np.array([], dtype=np.int64)
            for cid, parts in enumerate(splits)
        }
        min_client_size = min(len(v) for v in result.values())
        if min_client_size >= min_size:
            return result

    return _iid_split(len(labels), num_clients, rng)


def simulate_split(
    labels: np.ndarray,
    num_clients: int,
    split_type: str,
    seed: int,
    min_classes: int = 2,
    dirichlet_alpha: float = 0.3,
    unbalanced_keep_min: float = 0.5,
) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    split_type = split_type.lower()

    if split_type == "iid":
        return _iid_split(len(labels), num_clients, rng)
    if split_type == "unbalanced":
        return _unbalanced_split(len(labels), num_clients, rng, keep_min=unbalanced_keep_min)
    if split_type in {"patho", "pathological"}:
        return _pathological_split(labels, num_clients, min_classes=min_classes, rng=rng)
    if split_type in {"diri", "dirichlet"}:
        return _dirichlet_split(
            labels,
            num_clients,
            alpha=dirichlet_alpha,
            min_size=2,
            rng=rng,
        )
    raise ValueError(f"Unsupported split_type: {split_type}")


def split_subset_for_client(
    raw_train: Dataset,
    sample_indices: np.ndarray,
    client_id: int,
    test_size: float,
    seed: int,
    raw_targets: np.ndarray | None = None,
):
    sample_indices = np.asarray(sample_indices, dtype=np.int64)
    rng = np.random.default_rng(seed + client_id)
    sample_indices = rng.permutation(sample_indices)

    n = len(sample_indices)
    if not (test_size > 0 and n > 1):
        train_idx = sample_indices
        test_idx = np.asarray([], dtype=np.int64)
    else:
        # Class-aware split: keep at least one example per class in train when possible.
        targets_all = raw_targets if raw_targets is not None else extract_targets(raw_train)
        local_targets = targets_all[sample_indices]
        train_parts = []
        test_parts = []
        for cls in np.unique(local_targets):
            cls_mask = local_targets == cls
            cls_indices = sample_indices[cls_mask]
            cls_indices = rng.permutation(cls_indices)
            if len(cls_indices) <= 1:
                train_parts.append(cls_indices)
                continue
            cls_n_test = int(len(cls_indices) * float(test_size))
            cls_n_test = max(1, min(cls_n_test, len(cls_indices) - 1))
            test_parts.append(cls_indices[:cls_n_test])
            train_parts.append(cls_indices[cls_n_test:])
        train_idx = (
            np.concatenate(train_parts).astype(np.int64)
            if train_parts
            else np.asarray([], dtype=np.int64)
        )
        test_idx = (
            np.concatenate(test_parts).astype(np.int64)
            if test_parts
            else np.asarray([], dtype=np.int64)
        )
        train_idx = rng.permutation(train_idx) if len(train_idx) > 0 else train_idx
        test_idx = rng.permutation(test_idx) if len(test_idx) > 0 else test_idx

    train_subset = Subset(raw_train, train_idx.tolist())
    test_subset = (
        Subset(raw_train, test_idx.tolist())
        if len(test_idx) > 0
        else Subset(raw_train, [])
    )

    train_targets = extract_targets(train_subset) if len(train_subset) > 0 else np.array([], dtype=np.int64)
    test_targets = extract_targets(test_subset) if len(test_subset) > 0 else np.array([], dtype=np.int64)
    train_subset.targets = torch.from_numpy(train_targets).long()
    test_subset.targets = torch.from_numpy(test_targets).long()
    return train_subset, test_subset


def clientize_raw_dataset(
    raw_train: Dataset,
    args: DatasetArgs,
):
    targets = extract_targets(raw_train)
    split_type = str(args.split_type).strip().lower()
    if split_type == "pre":
        client_indices = _predefined_client_split_indices(raw_train=raw_train, args=args)
    else:
        client_indices = simulate_split(
            labels=targets,
            num_clients=int(args.num_clients),
            split_type=str(args.split_type),
            seed=int(args.seed),
            min_classes=int(args.min_classes),
            dirichlet_alpha=float(args.dirichlet_alpha),
            unbalanced_keep_min=float(args.unbalanced_keep_min),
        )

    client_datasets = []
    for cid in sorted(int(k) for k in client_indices.keys()):
        train_ds, test_ds = split_subset_for_client(
            raw_train=raw_train,
            sample_indices=client_indices[cid],
            client_id=cid,
            test_size=float(args.test_size),
            seed=int(args.seed),
            raw_targets=targets,
        )
        client_datasets.append((train_ds, test_ds))

    return client_datasets


def _predefined_client_split_indices(
    raw_train: Dataset,
    args: DatasetArgs,
) -> dict[int, np.ndarray]:
    source = str(getattr(args, "pre_source", "")).strip()
    pre_index = _safe_int(getattr(args, "pre_index", -1), -1)
    if source == "" and pre_index < 0:
        raise ValueError(
            "split.type='pre' requires split.configs.pre_source (key/column name) "
            "or split.configs.pre_index (tuple/list position)."
        )
    values = _extract_pre_source_values(
        raw_train=raw_train,
        source=source,
        pre_index=pre_index,
    )
    if values is None:
        raise ValueError(
            f"Unable to extract pre split source '{source or f'index {pre_index}'}' from dataset. "
            "For HF, ensure split.configs.pre_source matches an existing column."
        )
    if len(values) != len(raw_train):
        raise ValueError(
            f"Pre split source '{source}' length mismatch: {len(values)} vs {len(raw_train)}."
        )
    unique_values = sorted({str(v) for v in values})
    if not unique_values:
        raise ValueError("Pre split source produced no client identifiers.")

    requested_num = _safe_int(getattr(args, "num_clients", 0), 0)
    infer_num = _safe_bool(getattr(args, "pre_infer_num_clients", False), False)
    if infer_num or requested_num <= 0:
        selected_values = unique_values
    else:
        if requested_num > len(unique_values):
            raise ValueError(
                f"Requested train.num_clients={requested_num}, but pre source '{source}' "
                f"contains only {len(unique_values)} unique client ids."
            )
        selected_values = unique_values[:requested_num]

    selected_set = set(selected_values)
    index_bins: dict[str, list[int]] = {k: [] for k in selected_values}
    for idx, raw in enumerate(values):
        key = str(raw)
        if key in selected_set:
            index_bins[key].append(int(idx))

    non_empty_keys = [k for k, arr in index_bins.items() if len(arr) > 0]
    if not non_empty_keys:
        raise ValueError("Pre split produced no non-empty client subsets.")

    result: dict[int, np.ndarray] = {}
    for cid, key in enumerate(non_empty_keys):
        result[int(cid)] = np.asarray(index_bins[key], dtype=np.int64)
    args.num_clients = len(result)
    return result


def _extract_pre_source_values(raw_train: Dataset, source: str, pre_index: int):
    if source != "":
        direct = getattr(raw_train, source, None)
        if direct is not None:
            if torch.is_tensor(direct):
                return direct.detach().cpu().numpy().reshape(-1).tolist()
            arr = np.asarray(direct)
            if arr.size > 0:
                return arr.reshape(-1).tolist()

        if hasattr(raw_train, "metadata"):
            meta = getattr(raw_train, "metadata", None)
            if isinstance(meta, dict) and source in meta:
                arr = np.asarray(meta[source])
                if arr.size > 0:
                    return arr.reshape(-1).tolist()

        values = []
        for i in range(len(raw_train)):
            item = raw_train[i]
            if isinstance(item, dict):
                if source not in item:
                    return None
                values.append(item[source])
                continue
            return None
        return values

    if pre_index < 0:
        return None
    values = []
    for i in range(len(raw_train)):
        item = raw_train[i]
        if not isinstance(item, (tuple, list)):
            return None
        if pre_index >= len(item):
            return None
        values.append(item[int(pre_index)])
    return values


def set_common_metadata(
    args: DatasetArgs,
    client_datasets,
    raw_train: Dataset | None = None,
):
    args.num_clients = len(client_datasets)

    shape_source = raw_train if raw_train is not None and len(raw_train) > 0 else _first_nonempty_train_dataset(client_datasets)
    args.input_shape = infer_input_shape(shape_source) if shape_source is not None else (1,)

    if raw_train is not None and len(raw_train) > 0:
        args.num_classes = infer_num_classes(raw_train)
    else:
        train_targets = concat_targets(
            train_ds
            for train_ds in (
                _train_dataset_from_entry(entry) for entry in client_datasets
            )
            if train_ds is not None
        )
        args.num_classes = int(np.unique(train_targets).size) if train_targets.size > 0 else 0

    if getattr(args, "in_channels", None) is None:
        if len(args.input_shape) == 1:
            args.in_channels = 1
        elif len(args.input_shape) > 1:
            args.in_channels = int(args.input_shape[0])
    return args
