from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from appfl.loader.data.data_utils import _coerce_target_tensor, extract_targets


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


def _is_classification_targets(targets: np.ndarray) -> bool:
    arr = np.asarray(targets)
    if arr.size == 0:
        return True
    if np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.bool_):
        return True
    if np.issubdtype(arr.dtype, np.floating):
        return False
    return False


def _iid_partition(
    num_samples: int,
    num_clients: int,
    rng: np.random.Generator,
) -> dict[int, np.ndarray]:
    perm = rng.permutation(num_samples)
    chunks = np.array_split(perm, num_clients)
    return {cid: chunk.astype(np.int64) for cid, chunk in enumerate(chunks)}


def _unbalanced_partition(
    num_samples: int,
    num_clients: int,
    rng: np.random.Generator,
    keep_min: float,
) -> dict[int, np.ndarray]:
    base = _iid_partition(num_samples, num_clients, rng)
    out: dict[int, np.ndarray] = {}
    for cid, indices in base.items():
        if len(indices) <= 1:
            out[cid] = indices
            continue
        keep_ratio = rng.uniform(keep_min, 1.0)
        keep_count = max(1, int(len(indices) * keep_ratio))
        out[cid] = indices[:keep_count]
    return out


def _pathological_partition(
    labels: np.ndarray,
    num_clients: int,
    pathological_min_classes: int,
    rng: np.random.Generator,
) -> dict[int, np.ndarray]:
    cap = max(1, int(pathological_min_classes))
    classes = np.asarray(sorted(np.unique(labels)), dtype=np.int64)
    if classes.size == 0:
        return {cid: np.array([], dtype=np.int64) for cid in range(num_clients)}

    total_slots = int(num_clients) * int(cap)
    if total_slots < int(classes.size):
        raise ValueError(
            "pathological partition cannot cover all classes with current settings: "
            f"num_clients({num_clients}) * pathological_min_classes({cap}) "
            f"< num_classes({int(classes.size)})."
        )

    client_slots = rng.permutation(
        np.repeat(np.arange(num_clients, dtype=np.int64), cap)
    )
    class_to_clients: dict[int, list[int]] = {int(cls): [] for cls in classes.tolist()}
    client_class_sets = [set() for _ in range(num_clients)]

    shuffled_classes = rng.permutation(classes)
    for cls, cid in zip(
        shuffled_classes.tolist(), client_slots[: classes.size].tolist()
    ):
        c = int(cid)
        y = int(cls)
        client_class_sets[c].add(y)
        class_to_clients[y].append(c)

    for cid in client_slots[classes.size :].tolist():
        c = int(cid)
        available = [
            int(cls) for cls in classes.tolist() if int(cls) not in client_class_sets[c]
        ]
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
            raise RuntimeError(
                f"pathological partition internal error: class {y} has no owner."
            )
        parts = np.array_split(cls_idx, len(owners))
        for cid, part in zip(owners, parts):
            if len(part) > 0:
                out[int(cid)].append(part.astype(np.int64))

    result = {
        cid: np.concatenate(parts).astype(np.int64)
        if parts
        else np.array([], dtype=np.int64)
        for cid, parts in out.items()
    }

    covered = set()
    for cid, idx in result.items():
        if idx.size == 0:
            continue
        uniq = set(np.unique(labels[idx]).astype(np.int64).tolist())
        if len(uniq) > cap:
            raise RuntimeError(
                f"pathological partition produced {len(uniq)} classes for client {cid}, cap={cap}."
            )
        covered.update(int(v) for v in uniq)
    expected = {int(v) for v in classes.tolist()}
    if covered != expected:
        raise RuntimeError(
            "pathological partition failed to cover all classes across clients."
        )

    return result


def _dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    min_size: int,
    rng: np.random.Generator,
) -> dict[int, np.ndarray]:
    classes = np.unique(labels)
    alpha = max(alpha, 1e-3)
    total = int(labels.shape[0])
    avg_client_size = float(total) / float(num_clients)

    while True:
        partitions = [[] for _ in range(num_clients)]
        for cls in classes:
            cls_idx = np.where(labels == cls)[0]
            rng.shuffle(cls_idx)
            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.asarray(
                [
                    p * (sum(len(part) for part in partitions[cid]) < avg_client_size)
                    for cid, p in enumerate(proportions)
                ],
                dtype=float,
            )
            if float(proportions.sum()) <= 0.0:
                proportions = np.repeat(1.0 / float(num_clients), num_clients)
            else:
                proportions = proportions / float(proportions.sum())
            cut_points = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
            class_partitions = np.split(cls_idx, cut_points)
            for cid, part in enumerate(class_partitions):
                partitions[cid].append(part)

        result = {}
        for cid, parts in enumerate(partitions):
            client_idx = (
                np.concatenate(parts).astype(np.int64, copy=False)
                if parts
                else np.array([], dtype=np.int64)
            )
            rng.shuffle(client_idx)
            result[cid] = client_idx

        min_client_size = min(len(v) for v in result.values())
        if min_client_size >= min_size:
            return result


def simulate_partition(
    labels: np.ndarray,
    num_clients: int,
    partition_type: str,
    seed: int,
    pathological_min_classes: int = 2,
    dirichlet_alpha: float = 0.3,
    dirichlet_min_size: int = 2,
    unbalanced_keep_min: float = 0.5,
) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    partition_type = partition_type.lower()

    if partition_type == "iid":
        return _iid_partition(len(labels), num_clients, rng)
    if partition_type == "unbalanced":
        return _unbalanced_partition(
            len(labels), num_clients, rng, keep_min=unbalanced_keep_min
        )
    if partition_type in {"patho", "pathological"}:
        return _pathological_partition(
            labels,
            num_clients,
            pathological_min_classes=pathological_min_classes,
            rng=rng,
        )
    if partition_type in {"diri", "dirichlet"}:
        return _dirichlet_partition(
            labels,
            num_clients,
            alpha=dirichlet_alpha,
            min_size=max(1, int(dirichlet_min_size)),
            rng=rng,
        )
    raise ValueError(f"Unsupported partition_type: {partition_type}")


def split_client_dataset(
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
        targets_all = (
            raw_targets if raw_targets is not None else extract_targets(raw_train)
        )
        if _is_classification_targets(targets_all):
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
        else:
            n_test = int(len(sample_indices) * float(test_size))
            n_test = max(1, min(n_test, len(sample_indices) - 1))
            test_idx = sample_indices[:n_test]
            train_idx = sample_indices[n_test:]

    train_subset = Subset(raw_train, train_idx.tolist())
    test_subset = (
        Subset(raw_train, test_idx.tolist())
        if len(test_idx) > 0
        else Subset(raw_train, [])
    )

    train_targets = (
        extract_targets(train_subset)
        if len(train_subset) > 0
        else np.array([], dtype=np.int64)
    )
    test_targets = (
        extract_targets(test_subset)
        if len(test_subset) > 0
        else np.array([], dtype=np.int64)
    )
    train_subset.targets = _coerce_target_tensor(train_targets)
    test_subset.targets = _coerce_target_tensor(test_targets)
    return train_subset, test_subset


def partition_raw_dataset(
    raw_train: Dataset,
    config,
):
    targets = extract_targets(raw_train)
    partition_type = (
        str(getattr(config, "partition_type", getattr(config, "split_type", "iid")))
        .strip()
        .lower()
    )
    if partition_type == "pre":
        client_indices = _predefined_client_partition_indices(
            raw_train=raw_train, config=config
        )
    else:
        client_indices = simulate_partition(
            labels=targets,
            num_clients=int(getattr(config, "num_clients")),
            partition_type=partition_type,
            seed=int(getattr(config, "data_seed", 42)),
            pathological_min_classes=int(
                getattr(config, "pathological_min_classes", 2)
            ),
            dirichlet_alpha=float(getattr(config, "dirichlet_alpha", 0.3)),
            dirichlet_min_size=int(getattr(config, "dirichlet_min_size", 2)),
            unbalanced_keep_min=float(getattr(config, "unbalanced_keep_min", 0.5)),
        )

    client_datasets = []
    for cid in sorted(int(k) for k in client_indices.keys()):
        train_ds, test_ds = split_client_dataset(
            raw_train=raw_train,
            sample_indices=client_indices[cid],
            client_id=cid,
            test_size=float(getattr(config, "test_size", 0.2)),
            seed=int(getattr(config, "data_seed", 42)),
            raw_targets=targets,
        )
        client_datasets.append((train_ds, test_ds))

    return client_datasets


def _predefined_client_partition_indices(
    raw_train: Dataset,
    config,
) -> dict[int, np.ndarray]:
    source = str(getattr(config, "pre_source", "")).strip()
    pre_index = _safe_int(getattr(config, "pre_index", -1), -1)
    if source == "" and pre_index < 0:
        raise ValueError(
            "partition='pre' requires partition_kwargs.pre_source (key/column name) "
            "or partition_kwargs.pre_index (tuple/list position)."
        )
    values = _extract_pre_source_values(
        raw_train=raw_train,
        source=source,
        pre_index=pre_index,
    )
    if values is None:
        raise ValueError(
            f"Unable to extract pre-partition source '{source or f'index {pre_index}'}' from dataset. "
            "For HF, ensure partition_kwargs.pre_source matches an existing column."
        )
    if len(values) != len(raw_train):
        raise ValueError(
            f"Pre-partition source '{source}' length mismatch: {len(values)} vs {len(raw_train)}."
        )
    unique_values = sorted({str(v) for v in values})
    if not unique_values:
        raise ValueError("Pre-partition source produced no client identifiers.")

    requested_num = _safe_int(getattr(config, "num_clients", 0), 0)
    infer_num = _safe_bool(getattr(config, "pre_infer_num_clients", False), False)
    if infer_num or requested_num <= 0:
        selected_values = unique_values
    else:
        if requested_num > len(unique_values):
            raise ValueError(
                f"Requested num_clients={requested_num}, but pre-partition source '{source}' "
                f"contains only {len(unique_values)} unique client ids."
            )
        selected_values = unique_values[:requested_num]

    selected_set = set(selected_values)
    index_bins: dict[str, list[int]] = {k: [] for k in selected_values}
    for idx, raw in enumerate(values):
        key = str(raw)
        if key in selected_set:
            index_bins[key].append(int(idx))

    min_samples = max(0, _safe_int(getattr(config, "pre_min_samples_per_client", 0), 0))
    non_empty_keys = [
        k for k, arr in index_bins.items() if len(arr) >= max(1, min_samples or 1)
    ]
    if not non_empty_keys:
        if min_samples > 0:
            raise ValueError(
                "Pre-partition produced no client subsets meeting "
                f"partition_kwargs.pre_min_samples_per_client={min_samples}."
            )
        raise ValueError("Pre-partition produced no non-empty client subsets.")

    result: dict[int, np.ndarray] = {}
    for cid, key in enumerate(non_empty_keys):
        result[int(cid)] = np.asarray(index_bins[key], dtype=np.int64)
    config.num_clients = len(result)
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
