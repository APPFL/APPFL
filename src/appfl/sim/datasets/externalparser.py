from __future__ import annotations

from collections import Counter
from io import BytesIO
import logging
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image

from appfl.sim.datasets.common import (
    BasicTensorDataset,
    clientize_raw_dataset,
    finalize_dataset_outputs,
    infer_num_classes,
    make_load_tag,
    resolve_dataset_logger,
)


logger = logging.getLogger(__name__)


def _as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    return str(value)


def _normalize_labels(raw_labels: Iterable[Any]) -> torch.Tensor:
    values = list(raw_labels)
    if not values:
        return torch.zeros(0, dtype=torch.long)

    if all(isinstance(v, (int, np.integer)) for v in values):
        return torch.tensor([int(v) for v in values], dtype=torch.long)

    mapping = {label: i for i, label in enumerate(sorted({str(v) for v in values}))}
    return torch.tensor([mapping[str(v)] for v in values], dtype=torch.long)


def _pick_label_key(columns: list[str], args) -> str:
    user_key = str(getattr(args, "ext_label_key", "")).strip()
    if user_key:
        if user_key not in columns:
            raise ValueError(
                f"dataset.configs.label_key='{user_key}' not found in dataset columns: {columns}"
            )
        return user_key

    for cand in ["label", "labels", "target", "y", "class"]:
        if cand in columns:
            return cand

    raise ValueError(
        f"Unable to infer label column. Set dataset.configs.label_key explicitly. Available columns: {columns}"
    )


def _pick_feature_key(columns: list[str], label_key: str, args) -> str:
    user_key = str(getattr(args, "ext_feature_key", "")).strip()
    if user_key:
        if user_key not in columns:
            raise ValueError(
                f"dataset.configs.feature_key='{user_key}' not found in dataset columns: {columns}"
            )
        return user_key

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


def _tokenizer_from_args(args):
    tokenizer = None
    if bool(getattr(args, "use_model_tokenizer", False)):
        try:
            from transformers import AutoTokenizer

            model_name = str(getattr(args, "model_name", "")).strip()
            model_name = model_name if "/" in model_name else ""

            tokenizer_name = model_name or "bert-base-uncased"
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception:
            tokenizer = None
    return tokenizer


def _encode_text_features(texts: list[str], args) -> tuple[torch.Tensor, int]:
    seq_len = int(getattr(args, "seq_len", 128))
    tokenizer = _tokenizer_from_args(args)

    if tokenizer is not None:
        ids = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=seq_len,
            return_tensors="pt",
        )["input_ids"].long()
        return ids, int(tokenizer.vocab_size)

    vocab_size = int(getattr(args, "num_embeddings", 10000))
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
        arr = np.asarray(value)
    elif isinstance(value, dict) and "bytes" in value:
        arr = np.asarray(Image.open(BytesIO(value["bytes"])))
    else:
        arr = np.asarray(value)

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

    chw = np.transpose(hwc, (2, 0, 1))
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


def _rows_to_tensor_dataset(rows, feature_key: str, label_key: str, args, name: str):
    row_count = len(rows)
    split_type = str(getattr(args, "split_type", "")).strip().lower()
    pre_source = str(getattr(args, "pre_source", "")).strip()
    pre_values = [] if (split_type == "pre" and pre_source != "") else None
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
    for idx in range(row_count):
        row = rows[idx]
        features.append(row[feature_key])
        raw_labels.append(row[label_key])
        if pre_values is not None:
            if pre_source not in row:
                raise ValueError(
                    f"split.configs.pre_source='{pre_source}' not found in HF row columns."
                )
            pre_values.append(row[pre_source])
    labels = _normalize_labels(raw_labels)

    first = features[0]
    if isinstance(first, str) or (
        isinstance(first, (list, tuple)) and first and isinstance(first[0], str)
    ):
        texts = [_as_text(v) for v in features]
        x_tensor, vocab_size = _encode_text_features(texts, args)
        args.need_embedding = True
        args.seq_len = int(x_tensor.shape[1])
        args.num_embeddings = int(vocab_size)
        ds = BasicTensorDataset(x_tensor, labels, name=name)
        if pre_values is not None:
            setattr(ds, pre_source, np.asarray(pre_values, dtype=object))
        return ds

    if isinstance(first, Image.Image) or (
        isinstance(first, np.ndarray) and np.asarray(first).ndim in {2, 3}
    ):
        x_tensor = torch.stack([_to_image_tensor(v) for v in features], dim=0)
        args.need_embedding = False
        args.seq_len = None
        args.num_embeddings = None
        ds = BasicTensorDataset(x_tensor, labels, name=name)
        if pre_values is not None:
            setattr(ds, pre_source, np.asarray(pre_values, dtype=object))
        return ds

    if isinstance(first, dict) and "array" in first:
        nframes = int(getattr(args, "audio_num_frames", 16000))
        x_tensor = torch.stack([_to_audio_tensor(v, nframes) for v in features], dim=0)
        args.need_embedding = False
        args.seq_len = None
        args.num_embeddings = None
        ds = BasicTensorDataset(x_tensor, labels, name=name)
        if pre_values is not None:
            setattr(ds, pre_source, np.asarray(pre_values, dtype=object))
        return ds

    x_np = np.asarray(features)
    x_tensor = torch.as_tensor(x_np)
    if x_tensor.ndim == 1:
        x_tensor = x_tensor.unsqueeze(-1)
    if x_tensor.dtype in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        x_tensor = x_tensor.long()
    else:
        x_tensor = x_tensor.float()
    args.need_embedding = False
    args.seq_len = None
    args.num_embeddings = None
    ds = BasicTensorDataset(x_tensor, labels, name=name)
    if pre_values is not None:
        setattr(ds, pre_source, np.asarray(pre_values, dtype=object))
    return ds


def _fetch_hf_dataset(args, dataset_name: str):
    active_logger = resolve_dataset_logger(args, logger)
    tag = make_load_tag(dataset_name, benchmark="HF")
    active_logger.info("[%s] loading remote dataset splits.", tag)
    try:
        from datasets import DatasetDict, load_dataset
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "datasets (HuggingFace) is not installed. Install with: pip install datasets"
        ) from e

    config_name = str(getattr(args, "ext_config_name", "")).strip()
    train_split = str(getattr(args, "ext_train_split", "train")).strip()
    test_split = str(getattr(args, "ext_test_split", "test")).strip()

    kwargs: dict[str, Any] = {
        "cache_dir": str(getattr(args, "data_dir", "./data")),
    }
    try:
        if config_name:
            ds_obj = load_dataset(dataset_name, config_name, **kwargs)
        else:
            ds_obj = load_dataset(dataset_name, **kwargs)
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

        if test_split in ds_obj:
            test_hf = ds_obj[test_split]
        else:
            split = train_hf.train_test_split(
                test_size=float(getattr(args, "test_size", 0.2)),
                seed=int(getattr(args, "seed", 42)),
            )
            train_hf, test_hf = split["train"], split["test"]
    else:
        split = ds_obj.train_test_split(
            test_size=float(getattr(args, "test_size", 0.2)),
            seed=int(getattr(args, "seed", 42)),
        )
        train_hf, test_hf = split["train"], split["test"]

    if len(train_hf) == 0:
        raise ValueError(
            f"External HF dataset '{dataset_name}' has empty training split."
        )

    columns = list(getattr(train_hf, "column_names", [])) or list(train_hf[0].keys())
    label_key = _pick_label_key(columns, args)
    feature_key = _pick_feature_key(columns, label_key, args)
    if str(getattr(args, "split_type", "")).strip().lower() == "pre":
        pre_source = str(getattr(args, "pre_source", "")).strip()
        if pre_source == "":
            raise ValueError(
                "split.type='pre' requires split.configs.pre_source for HF backend."
            )
        if pre_source not in columns:
            raise ValueError(
                f"split.configs.pre_source='{pre_source}' not found in HF columns: {columns}"
            )

    raw_train = _rows_to_tensor_dataset(
        train_hf,
        feature_key=feature_key,
        label_key=label_key,
        args=args,
        name=f"[HF:{dataset_name}] TRAIN",
    )
    raw_test = _rows_to_tensor_dataset(
        test_hf,
        feature_key=feature_key,
        label_key=label_key,
        args=args,
        name=f"[HF:{dataset_name}] TEST",
    )

    client_datasets = clientize_raw_dataset(raw_train, args)
    active_logger.info("[%s] building federated client splits.", tag)
    client_datasets, server_dataset, dataset_meta = finalize_dataset_outputs(
        client_datasets=client_datasets,
        server_dataset=raw_test,
        dataset_meta=args,
        raw_train=raw_train,
    )
    dataset_meta.num_classes = int(infer_num_classes(raw_train))
    active_logger.info(
        "[%s] finished loading (%d clients).", tag, int(dataset_meta.num_clients)
    )
    return client_datasets, server_dataset, dataset_meta
