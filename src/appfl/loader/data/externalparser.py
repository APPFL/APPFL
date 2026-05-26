from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from io import BytesIO
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image

from appfl.loader.data.data_utils import (
    BasicTensorDataset,
    finalize_dataset_outputs,
    infer_num_classes,
    make_load_tag,
    resolve_dataset_logger,
)
from appfl.loader.data.partition_utils import partition_raw_dataset


logger = logging.getLogger(__name__)


def _hf_cache_repo_dir(cache_dir: str, dataset_name: str) -> Path:
    repo_cache_name = "___".join(part for part in str(dataset_name).split("/") if part)
    return Path(str(cache_dir)).expanduser() / repo_cache_name


def _hf_prepared_cache_available(cache_dir: str, dataset_name: str) -> bool:
    if str(cache_dir).strip() == "":
        return False

    repo_cache_dir = _hf_cache_repo_dir(cache_dir, dataset_name)
    if not repo_cache_dir.exists():
        return False

    has_dataset_info = False
    has_arrow_shard = False
    for path in repo_cache_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name == "dataset_info.json":
            has_dataset_info = True
        elif path.suffix == ".arrow":
            has_arrow_shard = True
        if has_dataset_info and has_arrow_shard:
            return True
    return False


@contextmanager
def _temporary_hf_hub_offline():
    previous = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous


def _load_hf_dataset_with_cache_preference(
    *,
    load_dataset_fn,
    dataset_name: str,
    config_name: str,
    load_kwargs: Dict[str, Any],
    active_logger,
    tag: str,
):
    def load_hf_dataset():
        if config_name:
            return load_dataset_fn(dataset_name, config_name, **load_kwargs)
        return load_dataset_fn(dataset_name, **load_kwargs)

    cache_dir = str(load_kwargs.get("cache_dir", "")).strip()
    if not _hf_prepared_cache_available(cache_dir, dataset_name):
        return load_hf_dataset()

    repo_cache_dir = _hf_cache_repo_dir(cache_dir, dataset_name)
    active_logger.info(
        "[%s] detected prepared local HF cache at '%s'; trying offline load first.",
        tag,
        str(repo_cache_dir),
    )
    try:
        with _temporary_hf_hub_offline():
            return load_hf_dataset()
    except Exception as exc:
        active_logger.warning(
            "[%s] offline HF cache-first load failed (%s); retrying with normal load.",
            tag,
            exc,
        )
        return load_hf_dataset()


def _as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    return str(value)


def _normalize_labels(raw_labels: Iterable[Any], regression: bool = False) -> torch.Tensor:
    values = list(raw_labels)
    if not values:
        dtype = torch.float32 if regression else torch.long
        empty = torch.zeros(0, dtype=dtype)
        return empty.unsqueeze(-1) if regression else empty

    if regression:
        return torch.tensor([float(v) for v in values], dtype=torch.float32).unsqueeze(-1)

    if all(isinstance(v, (int, np.integer)) for v in values):
        return torch.tensor([int(v) for v in values], dtype=torch.long)

    mapping = {label: i for i, label in enumerate(sorted({str(v) for v in values}))}
    return torch.tensor([mapping[str(v)] for v in values], dtype=torch.long)


def _resolve_column_selector(
    columns: List[str],
    selector: str,
    field_name: str,
    *,
    allow_empty: bool = False,
) -> str:
    raw_text = "" if selector is None else str(selector)
    text = raw_text.strip()
    if text == "":
        if allow_empty:
            return ""
        raise ValueError(f"{field_name} cannot be empty.")

    if raw_text in columns:
        return raw_text

    special_indices = {
        "__first_column__": 0,
        "__second_column__": 1,
        "__second_last_column__": len(columns) - 2,
        "__last_column__": len(columns) - 1,
    }
    if text in special_indices:
        index = int(special_indices[text])
        if index < 0 or index >= len(columns):
            raise ValueError(
                f"{field_name}='{text}' is invalid for columns: {columns}"
            )
        return columns[index]

    if text in columns:
        return text

    normalized_matches = [col for col in columns if str(col).strip() == text]
    if len(normalized_matches) == 1:
        return normalized_matches[0]
    if len(normalized_matches) > 1:
        raise ValueError(
            f"{field_name}='{raw_text}' matched multiple dataset columns after "
            f"whitespace normalization: {normalized_matches}"
        )
    raise ValueError(
        f"{field_name}='{text}' not found in dataset columns: {columns}"
    )


def _pick_label_key(columns: List[str], config) -> str:
    user_key = str(getattr(config, "ext_label_key", ""))
    if user_key.strip():
        return _resolve_column_selector(columns, user_key, "dataset.configs.label_key")

    for cand in ["label", "labels", "target", "y", "class"]:
        if cand in columns:
            return cand

    raise ValueError(
        f"Unable to infer label column. Set dataset.configs.label_key explicitly. Available columns: {columns}"
    )


def _pick_feature_key(columns: List[str], label_key: str, config) -> str:
    user_key = str(getattr(config, "ext_feature_key", ""))
    if user_key.strip():
        return _resolve_column_selector(columns, user_key, "dataset.configs.feature_key")

    preferred = [
        "image",
        "img",
        "pixel_values",
        "text",
        "sentence",
        "content",
        "tokens",
        "audio",
        "waveform",
        "x",
        "input",
        "input_ids",
    ]
    for key in preferred:
        if key in columns and key != label_key:
            return key

    remaining = [c for c in columns if c != label_key]
    if not remaining:
        raise ValueError("No feature column available after removing label column.")
    return remaining[0]


def _tokenizer_from_config(config):
    tokenizer = None
    if bool(getattr(config, "use_model_tokenizer", False)):
        try:
            from transformers import AutoTokenizer

            model_name = str(getattr(config, "model_name", "")).strip()
            model_name = model_name if "/" in model_name else ""

            tokenizer_name = model_name or "bert-base-uncased"
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception:
            tokenizer = None
    return tokenizer


def _encode_text_features(texts: List[str], config) -> Tuple[torch.Tensor, int]:
    seq_len = int(getattr(config, "seq_len", 128))
    tokenizer = _tokenizer_from_config(config)

    if tokenizer is not None:
        ids = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=seq_len,
            return_tensors="pt",
        )["input_ids"].long()
        return ids, int(tokenizer.vocab_size)

    vocab_size = int(getattr(config, "num_embeddings", 10000))
    basic = [t.lower().strip() for txt in texts for t in txt.split()]
    counter = Counter(basic)

    vocab = {"<pad>": 0, "<unk>": 1}
    for token, _ in counter.most_common(max(2, vocab_size - 2)):
        vocab[token] = len(vocab)

    def encode(text: str):
        tokens = [vocab.get(tok.lower().strip(), 1) for tok in text.split()]
        if len(tokens) < seq_len:
            tokens += [0] * (seq_len - len(tokens))
        return tokens[:seq_len]

    ids = torch.tensor([encode(txt) for txt in texts], dtype=torch.long)
    return ids, len(vocab)


def _to_image_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, Image.Image):
        arr = np.array(value, copy=True)
    elif isinstance(value, dict) and "bytes" in value:
        arr = np.array(Image.open(BytesIO(value["bytes"])), copy=True)
    else:
        arr = np.array(value, copy=True)

    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"Unsupported image shape: {arr.shape}")

    # Normalize image channels to fixed RGB (3 channels) so mixed grayscale/RGB
    # datasets can be stacked safely.
    if arr.shape[-1] in {1, 3, 4}:  # HWC
        hwc = arr
    elif arr.shape[0] in {1, 3, 4}:  # CHW
        hwc = np.transpose(arr, (1, 2, 0))
    else:
        raise ValueError(f"Unsupported image channel layout: {arr.shape}")

    channels = int(hwc.shape[-1])
    if channels == 1:
        hwc = np.repeat(hwc, 3, axis=-1)
    elif channels >= 4:
        hwc = hwc[..., :3]

    chw = np.transpose(hwc, (2, 0, 1)).copy()
    tensor = torch.from_numpy(chw).float()
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    return tensor


def _to_audio_tensor(value: Any, num_frames: int) -> torch.Tensor:
    if isinstance(value, dict) and "array" in value:
        arr = np.asarray(value["array"], dtype=np.float32)
    else:
        arr = np.asarray(value, dtype=np.float32)

    if arr.ndim > 1:
        arr = arr.reshape(-1)
    if arr.shape[0] < num_frames:
        arr = np.pad(arr, (0, num_frames - arr.shape[0]))
    elif arr.shape[0] > num_frames:
        arr = arr[:num_frames]
    return torch.from_numpy(arr).unsqueeze(0)


def _resolve_pre_source(columns: List[str], config) -> str:
    source = str(getattr(config, "pre_source", ""))
    if source.strip():
        resolved = _resolve_column_selector(columns, source, "partition_kwargs.pre_source")
        config.pre_source = resolved
        return resolved

    pre_index = int(getattr(config, "pre_index", -1))
    if pre_index < 0:
        return ""
    if pre_index >= len(columns):
        raise ValueError(
            f"partition_kwargs.pre_index={pre_index} is out of range for HF columns: {columns}"
        )
    resolved = columns[pre_index]
    config.pre_source = resolved
    return resolved


def _rows_to_tensor_dataset(rows, feature_key: str, label_key: str, config, name: str):
    row_count = len(rows)
    partition_type = str(
        getattr(config, "partition_type", getattr(config, "split_type", ""))
    ).strip().lower()
    pre_source = str(getattr(config, "pre_source", "")).strip()
    pre_values = [] if (partition_type == "pre" and pre_source != "") else None
    if row_count == 0:
        ds = BasicTensorDataset(
            torch.zeros(0, 1, dtype=torch.float32),
            torch.zeros(0, dtype=torch.long),
            name=name,
        )
        if pre_values is not None:
            setattr(ds, pre_source, np.asarray([], dtype=object))
        return ds

    features = []
    raw_labels = []
    regression_target = bool(getattr(config, "regression_target", False))
    for idx in range(row_count):
        row = rows[idx]
        features.append(row[feature_key])
        raw_labels.append(row[label_key])
        if pre_values is not None:
            if pre_source not in row:
                raise ValueError(
                    f"partition_kwargs.pre_source='{pre_source}' not found in HF row columns."
                )
            pre_values.append(row[pre_source])
    labels = _normalize_labels(raw_labels, regression=regression_target)

    first = features[0]
    if isinstance(first, str) or (
        isinstance(first, (list, tuple)) and first and isinstance(first[0], str)
    ):
        texts = [_as_text(v) for v in features]
        x_tensor, vocab_size = _encode_text_features(texts, config)
        config.need_embedding = True
        config.seq_len = int(x_tensor.shape[1])
        config.num_embeddings = int(vocab_size)
        ds = BasicTensorDataset(x_tensor, labels, name=name)
        if pre_values is not None:
            setattr(ds, pre_source, np.asarray(pre_values, dtype=object))
        return ds

    if isinstance(first, Image.Image) or (
        isinstance(first, np.ndarray) and np.asarray(first).ndim in {2, 3}
    ):
        x_tensor = torch.stack([_to_image_tensor(v) for v in features], dim=0)
        config.need_embedding = False
        config.seq_len = None
        config.num_embeddings = None
        ds = BasicTensorDataset(x_tensor, labels, name=name)
        if pre_values is not None:
            setattr(ds, pre_source, np.asarray(pre_values, dtype=object))
        return ds

    if isinstance(first, dict) and "array" in first:
        nframes = int(getattr(config, "audio_num_frames", 16000))
        x_tensor = torch.stack([_to_audio_tensor(v, nframes) for v in features], dim=0)
        config.need_embedding = False
        config.seq_len = None
        config.num_embeddings = None
        ds = BasicTensorDataset(x_tensor, labels, name=name)
        if pre_values is not None:
            setattr(ds, pre_source, np.asarray(pre_values, dtype=object))
        return ds

    x_np = np.asarray(features)
    x_tensor = torch.as_tensor(x_np)
    if x_tensor.ndim == 1:
        x_tensor = x_tensor.unsqueeze(-1)
    if x_tensor.dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
        x_tensor = x_tensor.long()
    else:
        x_tensor = x_tensor.float()
    config.need_embedding = False
    config.seq_len = None
    config.num_embeddings = None
    ds = BasicTensorDataset(x_tensor, labels, name=name)
    if pre_values is not None:
        setattr(ds, pre_source, np.asarray(pre_values, dtype=object))
    return ds


def fetch_hf_dataset(config):
    dataset_name = str(getattr(config, "dataset_name", "")).strip()
    if dataset_name == "":
        raise ValueError("dataset_name must be set for the hf dataset backend.")

    active_logger = resolve_dataset_logger(config, logger)
    tag = make_load_tag(dataset_name, benchmark="HF")
    active_logger.info("[%s] loading remote dataset splits.", tag)
    try:
        from datasets import DatasetDict, load_dataset
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "datasets (HuggingFace) is not installed. Install with: pip install datasets"
        ) from e

    config_name = str(getattr(config, "ext_config_name", "")).strip()
    train_split = str(getattr(config, "ext_train_split", "train")).strip()
    test_split = str(getattr(config, "ext_test_split", "test")).strip()

    kwargs: Dict[str, Any] = {
        "cache_dir": str(getattr(config, "data_dir", "./data")),
    }
    try:
        ds_obj = _load_hf_dataset_with_cache_preference(
            load_dataset_fn=load_dataset,
            dataset_name=dataset_name,
            config_name=config_name,
            load_kwargs=kwargs,
            active_logger=active_logger,
            tag=tag,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load HuggingFace dataset '{dataset_name}'. "
            "Check network access and dataset name, or use a local/custom parser."
        ) from e

    if isinstance(ds_obj, DatasetDict):
        if train_split in ds_obj:
            train_hf = ds_obj[train_split]
        else:
            first = list(ds_obj.keys())[0]
            train_hf = ds_obj[first]

        if test_split and test_split in ds_obj:
            test_hf = ds_obj[test_split]
        else:
            split = train_hf.train_test_split(
                test_size=float(getattr(config, "test_size", 0.2)),
                seed=int(getattr(config, "data_seed", 42)),
            )
            train_hf, test_hf = split["train"], split["test"]
    else:
        split = ds_obj.train_test_split(
            test_size=float(getattr(config, "test_size", 0.2)),
            seed=int(getattr(config, "data_seed", 42)),
        )
        train_hf, test_hf = split["train"], split["test"]

    if len(train_hf) == 0:
        raise ValueError(f"External HF dataset '{dataset_name}' has empty training split.")

    columns = list(getattr(train_hf, "column_names", [])) or list(train_hf[0].keys())
    label_key = _pick_label_key(columns, config)
    feature_key = _pick_feature_key(columns, label_key, config)
    partition_type = str(
        getattr(config, "partition_type", getattr(config, "split_type", ""))
    ).strip().lower()
    if partition_type == "pre":
        pre_source = _resolve_pre_source(columns, config)
        if pre_source == "":
            raise ValueError(
                "partition='pre' requires partition_kwargs.pre_source or partition_kwargs.pre_index for HF backend."
            )

    raw_train = _rows_to_tensor_dataset(
        train_hf,
        feature_key=feature_key,
        label_key=label_key,
        config=config,
        name=f"[HF:{dataset_name}] TRAIN",
    )
    raw_test = _rows_to_tensor_dataset(
        test_hf,
        feature_key=feature_key,
        label_key=label_key,
        config=config,
        name=f"[HF:{dataset_name}] TEST",
    )

    client_datasets = partition_raw_dataset(raw_train, config)
    active_logger.info("[%s] building federated client splits.", tag)
    client_datasets, server_dataset, dataset_meta = finalize_dataset_outputs(
        client_datasets=client_datasets,
        server_dataset=raw_test,
        dataset_meta=config,
        raw_train=raw_train,
    )
    if bool(getattr(config, "regression_target", False)):
        dataset_meta.num_classes = 1
    else:
        dataset_meta.num_classes = int(infer_num_classes(raw_train))
    active_logger.info(
        "[%s] finished loading (%d clients).", tag, int(dataset_meta.num_clients)
    )
    return client_datasets, server_dataset, dataset_meta
