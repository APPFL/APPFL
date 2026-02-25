from __future__ import annotations
import ast
import random
from typing import Sequence
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, Subset, random_split
from appfl.sim.logger import ServerAgentFileLogger
from appfl.sim.misc.config_utils import _cfg_get, _cfg_set


def _parse_holdout_dataset_ratio(config: DictConfig) -> list[float] | None:
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


def _safe_split_lengths(n: int, ratios: list[float]) -> list[int]:
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


def _dataset_targets(dataset) -> np.ndarray | None:
    def _as_label_array(value) -> np.ndarray | None:
        if value is None:
            return None
        if torch.is_tensor(value):
            arr = value.detach().cpu().numpy()
        else:
            arr = np.asarray(value)
        if arr.size == 0:
            return np.asarray([], dtype=np.int64)
        return arr.reshape(-1).astype(np.int64, copy=False)

    def _subset_indices_array(indices, subset_len: int) -> np.ndarray | None:
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
        parts: list[np.ndarray] = []
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
    ratios: list[float],
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


def _normalize_client_tuple(
    entry,
) -> tuple[object | None, object | None, object | None]:
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
    logger: ServerAgentFileLogger | None = None,
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

    seed = int(_cfg_get(config, "experiment.seed", 0))
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
        if splits is None:
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


def _validate_loader_output(client_datasets, runtime_cfg: dict) -> None:
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


def _build_client_groups(
    config: DictConfig, num_clients: int
) -> tuple[list[int], list[int]]:
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

    rng = random.Random(int(_cfg_get(config, "experiment.seed", 42)) + 2026)
    shuffled = all_clients[:]
    rng.shuffle(shuffled)
    holdout = sorted(shuffled[:holdout_num])
    train_clients = sorted(cid for cid in all_clients if cid not in set(holdout))
    if not train_clients:
        return all_clients, []
    return train_clients, holdout


def _sample_train_clients(
    train_client_ids: list[int], num_sampled_clients: int
) -> list[int]:
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


## Algorithm-specific methods
def _validate_bandit_dataset_ratio(config: DictConfig) -> None:
    algorithm = str(_cfg_get(config, "algorithm.name", "fedavg")).strip().lower()
    scheduler_name = str(_cfg_get(config, "algorithm.scheduler", "")).strip().lower()
    is_bandit = algorithm in {"swucb", "swts"} or scheduler_name in {
        "swucbscheduler",
        "swtsscheduler",
    }
    if not is_bandit:
        return
    ratios = _parse_holdout_dataset_ratio(config)
    if ratios is None:
        raise ValueError(
            "For algorithm in {swucb, swts}, `eval.configs.dataset_ratio` is required "
            "and must include validation split, e.g. [80,10,10]."
        )
    if len(ratios) < 3:
        raise ValueError(
            "For algorithm in {swucb, swts}, `eval.configs.dataset_ratio` must have "
            "three entries (train/val/test), e.g. [80,10,10]."
        )


def _validate_algorithm_data_requirements(config: DictConfig) -> None:
    # Keep runner decoupled from algorithm-specific checks.
    # Extend this dispatcher as new algorithms introduce data/runtime constraints.
    _validate_bandit_dataset_ratio(config)
