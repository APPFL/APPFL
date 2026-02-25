from __future__ import annotations

import importlib
import hashlib
import json
import logging
import random
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from appfl.sim.datasets.common import (
    package_dataset_outputs,
    resolve_fixed_pool_clients,
    to_namespace,
)


_TEXT_DATASETS = {"shakespeare", "sent140", "reddit"}
_DEFAULT_LEAF_META = {
    "femnist": {"num_classes": 62, "need_embedding": False},
    "shakespeare": {"num_classes": 80, "need_embedding": True, "seq_len": 80, "num_embeddings": 80},
    "sent140": {"num_classes": 2, "need_embedding": True, "seq_len": 25, "num_embeddings": 400001},
    "celeba": {"num_classes": 2, "need_embedding": False},
    "reddit": {"num_classes": 10000, "need_embedding": True, "seq_len": 10, "num_embeddings": 10000},
}
_LEAF_SUPPORTED = set(_DEFAULT_LEAF_META.keys())

logger = logging.getLogger(__name__)


def _stable_hash_to_mod(value: str, mod: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()
    return int(digest[:16], 16) % max(1, int(mod))


class _LeafLoggerAdapter:
    """Adapter that normalizes APPFL custom loggers to stdlib-like logger methods."""

    def __init__(self, target):
        self._target = target

    def _fmt(self, msg: str, *args) -> str:
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


def _resolve_leaf_logger(args):
    candidate = getattr(args, "logger", None)
    if candidate is None:
        return logger
    # Preserve standard loggers, adapt custom APPFL loggers.
    if isinstance(candidate, logging.Logger):
        return candidate
    return _LeafLoggerAdapter(candidate)


def _as_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _has_json_files(directory: Path) -> bool:
    return directory.exists() and any(directory.glob("*.json"))


def _has_train_test_json(dataset_root: Path) -> bool:
    return _has_json_files(dataset_root / "train") and _has_json_files(dataset_root / "test")


def _is_leaf_ready(dataset_root: Path) -> bool:
    return _has_train_test_json(dataset_root) or _has_json_files(dataset_root / "all_data")


def _prepare_leaf_data(args, dataset_key: str) -> Path:
    data_root = Path(str(args.data_dir)).expanduser()
    dataset_root = data_root / dataset_key
    leaf_logger = _resolve_leaf_logger(args)
    if _is_leaf_ready(dataset_root):
        return dataset_root

    if dataset_key not in _LEAF_SUPPORTED:
        raise ValueError(
            f"Unsupported LEAF dataset `{dataset_key}`. "
            f"Supported: {sorted(_LEAF_SUPPORTED)}"
        )

    if not _as_bool(getattr(args, "download", True), default=True):
        raise FileNotFoundError(
            f"LEAF dataset not found at {dataset_root}. "
            "Set `download=true` to auto-download and preprocess."
        )

    from appfl.sim.datasets.leaf import download_data, postprocess_leaf

    dataset_root.mkdir(parents=True, exist_ok=True)
    raw_dir = dataset_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not any(raw_dir.iterdir()):
        leaf_logger.info("[LEAF-%s] raw artifacts missing; downloading.", dataset_key.upper())
        download_data(
            download_root=str(raw_dir),
            dataset_name=dataset_key,
            logger=leaf_logger,
        )

    if not _has_json_files(dataset_root / "all_data"):
        leaf_logger.info("[LEAF-%s] preprocessing raw artifacts.", dataset_key.upper())
        try:
            preprocess_mod = importlib.import_module(
                f"appfl.sim.datasets.leaf.preprocess.{dataset_key}"
            )
        except ModuleNotFoundError as exc:
            if exc.name == "pandas":
                raise ModuleNotFoundError(
                    "LEAF Sent140 preprocessing requires `pandas`. "
                    "Install it with: pip install pandas"
                ) from exc
            raise
        if hasattr(preprocess_mod, "logger"):
            preprocess_mod.logger = leaf_logger
        try:
            preprocess_mod.preprocess(str(data_root), logger=leaf_logger)
        except TypeError:
            preprocess_mod.preprocess(str(data_root))

    if not _has_train_test_json(dataset_root):
        leaf_logger.info("[LEAF-%s] building train/test client splits.", dataset_key.upper())
        postprocess_leaf(
            dataset_name=dataset_key,
            root=str(data_root),
            seed=int(getattr(args, "seed", 42)),
            raw_data_fraction=float(getattr(args, "leaf_raw_data_fraction", 1.0)),
            min_samples_per_clients=int(getattr(args, "leaf_min_samples_per_client", 2)),
            test_size=float(getattr(args, "test_size", 0.2)),
            logger=leaf_logger,
        )

    if not _is_leaf_ready(dataset_root):
        raise RuntimeError(
            f"LEAF preparation failed for `{dataset_key}` at {dataset_root}."
        )
    return dataset_root


class LeafClientDataset(Dataset):
    def __init__(
        self,
        dataset_key: str,
        split: str,
        user: str,
        records: Dict[str, List[Any]],
        label_to_idx: Dict[str, int],
        seq_len: int | None,
        num_embeddings: int | None,
        image_root: Path | None,
    ):
        self.dataset_key = dataset_key
        self.identifier = f"[LEAF-{dataset_key.upper()}] CLIENT<{user}> ({split})"
        self.x = list(records.get("x", []))
        raw_y = list(records.get("y", []))
        self.targets = torch.tensor(
            [label_to_idx[str(v)] for v in raw_y], dtype=torch.long
        )
        self.seq_len = int(seq_len) if seq_len is not None else None
        self.num_embeddings = int(num_embeddings) if num_embeddings is not None else None
        self.image_root = image_root
        self._text_inputs: torch.Tensor | None = None

        # Text datasets are pre-tokenized before per-client dataset construction.
        if self.dataset_key in _TEXT_DATASETS:
            if self.x:
                first = self.x[0]
                if (
                    isinstance(first, (list, tuple))
                    and all(isinstance(v, (int, np.integer)) for v in first)
                ):
                    self._text_inputs = torch.tensor(self.x, dtype=torch.long)
                else:
                    self._text_inputs = torch.stack(
                        [self._encode_text(v) for v in self.x], dim=0
                    )
            else:
                seq_len = int(self.seq_len or 32)
                self._text_inputs = torch.zeros((0, seq_len), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.x)

    def __repr__(self) -> str:
        return self.identifier

    def _encode_text(self, value: Any) -> torch.Tensor:
        seq_len = int(self.seq_len or 32)
        vocab = max(8, int(self.num_embeddings or 256))

        if isinstance(value, (list, tuple)) and value and all(
            isinstance(v, (int, np.integer)) for v in value
        ):
            ids = [int(v) % vocab for v in value]
        else:
            if isinstance(value, bytes):
                text = value.decode("utf-8", errors="ignore")
            elif isinstance(value, str):
                text = value
            elif isinstance(value, (list, tuple)):
                text = " ".join(str(v) for v in value)
            else:
                text = str(value)

            tokens = list(text) if self.dataset_key == "shakespeare" else text.split()
            ids = [_stable_hash_to_mod(str(tok), vocab) for tok in tokens]

        if len(ids) < seq_len:
            ids += [0] * (seq_len - len(ids))
        return torch.tensor(ids[:seq_len], dtype=torch.long)

    @staticmethod
    def _to_chw_float(arr: np.ndarray) -> torch.Tensor:
        if arr.ndim == 1:
            side = int(round(np.sqrt(arr.size)))
            if side * side == arr.size:
                arr = arr.reshape(side, side)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=0)
        elif arr.ndim == 3 and arr.shape[0] not in {1, 3}:
            arr = np.transpose(arr, (2, 0, 1))

        tensor = torch.from_numpy(arr).float()
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        return tensor

    def _load_image_from_path(self, rel_or_abs_path: str, rgb: bool) -> torch.Tensor:
        p = Path(rel_or_abs_path)
        if not p.is_absolute() and self.image_root is not None:
            candidate = self.image_root / rel_or_abs_path
            if candidate.exists():
                p = candidate
            else:
                p = self.image_root / Path(rel_or_abs_path).name
        if not p.exists():
            raise FileNotFoundError(f"LEAF image file not found: {p}")

        img = Image.open(p)
        img = img.convert("RGB" if rgb else "L")
        return self._to_chw_float(np.asarray(img))

    def _encode_image_like(self, value: Any, rgb: bool) -> torch.Tensor:
        if isinstance(value, str):
            return self._load_image_from_path(value, rgb=rgb)
        arr = np.asarray(value)
        return self._to_chw_float(arr)

    def __getitem__(self, index: int):
        xi = self.x[index]
        yi = self.targets[index]

        if self.dataset_key in _TEXT_DATASETS:
            if self._text_inputs is None:
                x = self._encode_text(xi)
            else:
                x = self._text_inputs[index]
        elif self.dataset_key == "celeba":
            x = self._encode_image_like(xi, rgb=True)
        elif self.dataset_key == "femnist":
            x = self._encode_image_like(xi, rgb=False)
        else:
            arr = np.asarray(xi)
            x = self._to_chw_float(arr) if arr.ndim in {1, 2, 3} else torch.tensor(xi)

        return x, yi


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _merge_leaf_json_dir(folder: Path) -> Dict[str, Any]:
    files = sorted([p for p in folder.glob("*.json") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No JSON files found in {folder}")

    merged = {"users": [], "num_samples": [], "user_data": {}}
    for fp in files:
        obj = _load_json(fp)
        users = obj.get("users", [])
        user_data = obj.get("user_data", {})
        num_samples = obj.get("num_samples", [])
        for idx, user in enumerate(users):
            if user not in user_data:
                continue
            if user in merged["user_data"]:
                continue
            merged["users"].append(user)
            merged["user_data"][user] = user_data[user]
            if idx < len(num_samples):
                merged["num_samples"].append(int(num_samples[idx]))
            else:
                merged["num_samples"].append(len(user_data[user].get("y", [])))
    return merged


def _sample_users_by_fraction(all_obj: Dict[str, Any], fraction: float, seed: int) -> List[str]:
    users = list(all_obj.get("users", []))
    if fraction >= 1.0 or not users:
        return users

    rng = random.Random(seed)
    rng.shuffle(users)

    total = sum(len(all_obj["user_data"][u].get("y", [])) for u in users)
    target = max(1, int(total * max(0.0, fraction)))

    selected = []
    cum = 0
    for user in users:
        selected.append(user)
        cum += len(all_obj["user_data"][user].get("y", []))
        if cum >= target:
            break
    return selected


def _split_from_all_data(
    dataset_key: str,
    all_obj: Dict[str, Any],
    test_size: float,
    seed: int,
    raw_data_fraction: float,
    min_samples_per_client: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    rng = random.Random(seed)
    users = _sample_users_by_fraction(all_obj, raw_data_fraction, seed)

    train_obj = {"users": [], "num_samples": [], "user_data": {}}
    test_obj = {"users": [], "num_samples": [], "user_data": {}}

    for user in users:
        records = all_obj["user_data"].get(user, {})
        x = list(records.get("x", []))
        y = list(records.get("y", []))

        n = min(len(x), len(y))
        if n < max(2, int(min_samples_per_client)):
            continue

        x = x[:n]
        y = y[:n]
        n_train = max(1, min(int((1.0 - test_size) * n), n - 1))

        if dataset_key == "shakespeare":
            train_idx = list(range(n_train))
            test_idx = list(range(n_train, n))
        else:
            train_idx = sorted(rng.sample(range(n), n_train))
            test_set = set(range(n)) - set(train_idx)
            test_idx = sorted(list(test_set))

        if not train_idx or not test_idx:
            continue

        train_obj["users"].append(user)
        test_obj["users"].append(user)

        train_obj["user_data"][user] = {
            "x": [x[i] for i in train_idx],
            "y": [y[i] for i in train_idx],
        }
        test_obj["user_data"][user] = {
            "x": [x[i] for i in test_idx],
            "y": [y[i] for i in test_idx],
        }
        train_obj["num_samples"].append(len(train_idx))
        test_obj["num_samples"].append(len(test_idx))

    return train_obj, test_obj


def _resolve_leaf_train_test(args, dataset_root: Path, dataset_key: str):
    train_dir = dataset_root / "train"
    test_dir = dataset_root / "test"

    if train_dir.exists() and test_dir.exists():
        return _merge_leaf_json_dir(train_dir), _merge_leaf_json_dir(test_dir)

    all_data_dir = dataset_root / "all_data"
    if not all_data_dir.exists():
        raise FileNotFoundError(
            f"LEAF dataset requires either train/test JSON directories or all_data under {dataset_root}"
        )

    all_obj = _merge_leaf_json_dir(all_data_dir)
    return _split_from_all_data(
        dataset_key=dataset_key,
        all_obj=all_obj,
        test_size=float(getattr(args, "test_size", 0.2)),
        seed=int(getattr(args, "seed", 42)),
        raw_data_fraction=float(getattr(args, "leaf_raw_data_fraction", 1.0)),
        min_samples_per_client=int(getattr(args, "leaf_min_samples_per_client", 2)),
    )


def _resolve_image_root(args, dataset_root: Path, dataset_key: str) -> Path | None:
    user_root = str(getattr(args, "leaf_image_root", "")).strip()
    if user_root:
        p = Path(user_root).expanduser()
        if p.exists():
            return p

    if dataset_key == "celeba":
        candidates = [
            dataset_root / "raw" / "img_align_celeba",
            dataset_root / "img_align_celeba",
            dataset_root / "images",
        ]
        for path in candidates:
            if path.exists():
                return path
    return None


def _build_label_vocab(train_obj: Dict[str, Any], test_obj: Dict[str, Any]) -> Dict[str, int]:
    labels = []
    for obj in [train_obj, test_obj]:
        for user in obj.get("users", []):
            labels.extend(obj["user_data"].get(user, {}).get("y", []))
    unique = sorted({str(v) for v in labels})
    return {label: idx for idx, label in enumerate(unique)}


def _text_to_token_ids(
    value: Any,
    *,
    dataset_key: str,
    seq_len: int,
    num_embeddings: int,
) -> List[int]:
    vocab = max(8, int(num_embeddings))
    if isinstance(value, (list, tuple)) and value and all(
        isinstance(v, (int, np.integer)) for v in value
    ):
        ids = [int(v) % vocab for v in value]
    else:
        if isinstance(value, bytes):
            text = value.decode("utf-8", errors="ignore")
        elif isinstance(value, str):
            text = value
        elif isinstance(value, (list, tuple)):
            text = " ".join(str(v) for v in value)
        else:
            text = str(value)
        tokens = list(text) if dataset_key == "shakespeare" else text.split()
        ids = [_stable_hash_to_mod(str(tok), vocab) for tok in tokens]
    if len(ids) < seq_len:
        ids += [0] * (seq_len - len(ids))
    return ids[:seq_len]


def _pretokenize_leaf_text_data(
    *,
    dataset_key: str,
    train_obj: Dict[str, Any],
    test_obj: Dict[str, Any],
    users: List[str],
    seq_len: int,
    num_embeddings: int,
) -> None:
    if dataset_key not in _TEXT_DATASETS:
        return
    for obj in (train_obj, test_obj):
        user_data = obj.get("user_data", {})
        for user in users:
            records = user_data.get(user, None)
            if not isinstance(records, dict):
                continue
            raw_x = list(records.get("x", []))
            records["x"] = [
                _text_to_token_ids(
                    item,
                    dataset_key=dataset_key,
                    seq_len=int(seq_len),
                    num_embeddings=int(num_embeddings),
                )
                for item in raw_x
            ]


def fetch_leaf(args):
    """LEAF parser adapted from AAggFF processing flow with compact implementation."""
    args = to_namespace(args)
    leaf_logger = _resolve_leaf_logger(args)
    split_type = str(getattr(args, "split_type", "pre")).strip().lower()
    if split_type != "pre":
        raise ValueError(
            "For dataset.backend=leaf, split.type must be exactly `pre`."
        )
    dataset_key = str(args.dataset_name).strip().lower()
    leaf_logger.info("[LEAF-%s] load processed dataset.", dataset_key.upper())
    dataset_root = _prepare_leaf_data(args, dataset_key)

    train_obj, test_obj = _resolve_leaf_train_test(args, dataset_root, dataset_key)
    if not train_obj.get("users"):
        raise ValueError(f"No LEAF users available after processing: {dataset_root}")

    users = resolve_fixed_pool_clients(
        available_clients=list(train_obj["users"]),
        args=args,
        prefix="leaf",
    )
    if not users:
        raise ValueError(
            "No LEAF clients selected after applying num_clients/subsampling constraints."
        )
    leaf_logger.info("[LEAF-%s] set up %d clients.", dataset_key.upper(), len(users))

    label_to_idx = _build_label_vocab(train_obj, test_obj)
    image_root = _resolve_image_root(args, dataset_root, dataset_key)

    defaults = _DEFAULT_LEAF_META.get(dataset_key, {})
    seq_len = int(getattr(args, "seq_len", defaults.get("seq_len", 32)))
    num_embeddings = int(
        getattr(args, "num_embeddings", defaults.get("num_embeddings", 10000))
    )
    _pretokenize_leaf_text_data(
        dataset_key=dataset_key,
        train_obj=train_obj,
        test_obj=test_obj,
        users=users,
        seq_len=seq_len,
        num_embeddings=num_embeddings,
    )

    client_datasets = []
    for cid, user in enumerate(users):
        tr_records = train_obj["user_data"].get(user, {"x": [], "y": []})
        te_records = test_obj["user_data"].get(user, {"x": [], "y": []})
        tr_ds = LeafClientDataset(
            dataset_key=dataset_key,
            split="train",
            user=str(user),
            records=tr_records,
            label_to_idx=label_to_idx,
            seq_len=seq_len,
            num_embeddings=num_embeddings,
            image_root=image_root,
        )
        te_ds = LeafClientDataset(
            dataset_key=dataset_key,
            split="test",
            user=str(user),
            records=te_records,
            label_to_idx=label_to_idx,
            seq_len=seq_len,
            num_embeddings=num_embeddings,
            image_root=image_root,
        )
        client_datasets.append((tr_ds, te_ds))

    args.num_clients = len(client_datasets)
    args.num_classes = (
        len(label_to_idx) if label_to_idx else int(defaults.get("num_classes", 0))
    )

    args.need_embedding = bool(defaults.get("need_embedding", dataset_key in _TEXT_DATASETS))
    if args.need_embedding:
        args.seq_len = seq_len
        args.num_embeddings = num_embeddings
    else:
        args.seq_len = None
        args.num_embeddings = None

    first_train = next((tr for tr, _ in client_datasets if len(tr) > 0), None)
    if first_train is not None:
        x0, _ = first_train[0]
        args.input_shape = tuple(getattr(x0, "shape", (1,)))
    else:
        args.input_shape = (1,)

    if len(args.input_shape) > 1:
        args.in_channels = int(args.input_shape[0])
    else:
        args.in_channels = 1

    leaf_logger.info(
        "[LEAF-%s] finished loading (%d clients).",
        dataset_key.upper(),
        int(args.num_clients),
    )
    return package_dataset_outputs(
        client_datasets=client_datasets,
        server_dataset=None,
        dataset_meta=args,
    )
