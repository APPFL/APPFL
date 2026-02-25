from __future__ import annotations

from collections import Counter
import logging
from pathlib import Path

import torch

from appfl.sim.datasets.common import (
    BasicTensorDataset,
    clientize_raw_dataset,
    finalize_dataset_outputs,
    infer_num_classes,
    make_load_tag,
    resolve_dataset_logger,
    to_namespace,
)


logger = logging.getLogger(__name__)


def _tokenizer_from_args(args):
    tokenizer = None
    if bool(args.use_model_tokenizer):
        try:
            from transformers import AutoTokenizer

            model_name = str(getattr(args, "model_name", "")).strip()
            model_name = model_name if "/" in model_name else ""
            tokenizer_name = model_name or "bert-base-uncased"
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception:
            tokenizer = None
    return tokenizer


def _read_torchtext_split(ds_fn, root: str, split_name: str):
    # torchtext API varies significantly across versions.
    # Newer API: ds_fn(root=..., split=\"train\"|\"test\")
    # Older API (e.g., torchtext 0.6): ds_fn(root=...) -> (train_iter, test_iter)
    try:
        data_iter = ds_fn(root=root, split=split_name)
        return list(data_iter)
    except TypeError:
        pass

    try:
        both = ds_fn(root=root)
    except TypeError:
        both = ds_fn()

    if isinstance(both, tuple) and len(both) >= 2:
        idx = 0 if split_name == "train" else 1
        return list(both[idx])
    if split_name == "train":
        return list(both)

    raise ValueError(
        "Unable to obtain torchtext test split with this torchtext version. "
        "Please provide dataset with explicit train/test support."
    )


def _unpack_rows(rows):
    labels, texts = [], []
    for row in rows:
        if isinstance(row, tuple) and len(row) >= 2:
            label, text = row[0], row[1]
        elif isinstance(row, dict):
            label, text = row.get("label", 0), row.get("text", "")
        else:
            raise ValueError("Unsupported torchtext row format")
        labels.append(label)
        texts.append(text)
    return labels, texts


def fetch_torchtext_dataset(args):
    args = to_namespace(args)
    active_logger = resolve_dataset_logger(args, logger)
    tag = make_load_tag(str(args.dataset_name), benchmark="TORCHTEXT")
    active_logger.info("[%s] resolving dataset class.", tag)
    try:
        import torchtext
    except Exception as e:  # pragma: no cover
        raise RuntimeError("torchtext is not installed.") from e

    data_root = Path(str(args.data_dir)).expanduser()
    data_root.mkdir(parents=True, exist_ok=True)
    active_logger.info("[%s] reading train/test splits.", tag)

    if not hasattr(torchtext.datasets, args.dataset_name):
        raise ValueError(f"Unknown torchtext dataset: {args.dataset_name}")

    ds_fn = getattr(torchtext.datasets, args.dataset_name)
    train_rows = _read_torchtext_split(ds_fn, str(data_root), "train")
    test_rows = _read_torchtext_split(ds_fn, str(data_root), "test")

    tr_labels_raw, tr_texts = _unpack_rows(train_rows)
    te_labels_raw, te_texts = _unpack_rows(test_rows)

    label_vocab = {
        l: i for i, l in enumerate(sorted(set(tr_labels_raw + te_labels_raw)))
    }
    tr_labels = torch.tensor([label_vocab[l] for l in tr_labels_raw], dtype=torch.long)
    te_labels = torch.tensor([label_vocab[l] for l in te_labels_raw], dtype=torch.long)

    tokenizer = _tokenizer_from_args(args)
    seq_len = int(args.seq_len)

    if tokenizer is not None:
        tr_ids = tokenizer(
            tr_texts,
            truncation=True,
            padding="max_length",
            max_length=seq_len,
            return_tensors="pt",
        )["input_ids"]
        te_ids = tokenizer(
            te_texts,
            truncation=True,
            padding="max_length",
            max_length=seq_len,
            return_tensors="pt",
        )["input_ids"]
        args.num_embeddings = int(tokenizer.vocab_size)
    else:
        basic_tok = torchtext.data.utils.get_tokenizer("basic_english")
        counter = Counter()
        for txt in tr_texts:
            counter.update(basic_tok(txt))

        vocab = {"<pad>": 0, "<unk>": 1}
        for token, _ in counter.most_common(max(2, int(args.num_embeddings) - 2)):
            vocab[token] = len(vocab)

        def encode(text: str):
            tokens = [vocab.get(t, 1) for t in basic_tok(text)]
            if len(tokens) < seq_len:
                tokens = tokens + [0] * (seq_len - len(tokens))
            return tokens[:seq_len]

        tr_ids = torch.tensor([encode(t) for t in tr_texts], dtype=torch.long)
        te_ids = torch.tensor([encode(t) for t in te_texts], dtype=torch.long)
        args.num_embeddings = len(vocab)

    raw_train = BasicTensorDataset(
        tr_ids, tr_labels, name=f"[{args.dataset_name}] TRAIN"
    )
    raw_test = BasicTensorDataset(te_ids, te_labels, name=f"[{args.dataset_name}] TEST")
    active_logger.info("[%s] building federated client splits.", tag)

    client_datasets = clientize_raw_dataset(raw_train, args)
    args.need_embedding = True
    client_datasets, server_dataset, dataset_meta = finalize_dataset_outputs(
        client_datasets=client_datasets,
        server_dataset=raw_test,
        dataset_meta=args,
        raw_train=raw_train,
    )
    dataset_meta.num_classes = int(infer_num_classes(raw_train))
    dataset_meta.input_shape = tuple(tr_ids.shape[1:])
    dataset_meta.seq_len = (
        int(tr_ids.shape[1]) if tr_ids.ndim >= 2 else int(dataset_meta.seq_len)
    )
    dataset_meta.need_embedding = True
    active_logger.info(
        "[%s] finished loading (%d clients).", tag, int(dataset_meta.num_clients)
    )
    return client_datasets, server_dataset, dataset_meta
