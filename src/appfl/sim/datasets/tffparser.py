from __future__ import annotations

import hashlib
import logging

import numpy as np
import torch

from appfl.sim.datasets.common import (
    BasicTensorDataset,
    extract_targets,
    make_load_tag,
    package_dataset_outputs,
    resolve_dataset_logger,
    resolve_fixed_pool_clients,
    set_common_metadata,
    split_subset_for_client,
    to_namespace,
)


logger = logging.getLogger(__name__)


def _tf_to_numpy(value):
    if hasattr(value, "numpy"):
        return value.numpy()
    return value


def _stable_hash_to_mod(value: str, mod: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()
    return int(digest[:16], 16) % max(1, int(mod))


def fetch_tff_dataset(args):
    args = to_namespace(args)
    active_logger = resolve_dataset_logger(args, logger)
    split_type = str(getattr(args, "split_type", "pre")).strip().lower()
    if split_type != "pre":
        raise ValueError(
            "For dataset.backend=tff, split.type must be exactly `pre`."
        )
    try:
        import tensorflow_federated as tff
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "tensorflow_federated is not installed. "
            "On Python 3.14 this may fail due dependency constraints; use Python 3.10/3.11 environment for TFF datasets."
        ) from e

    if ":" in str(args.dataset_name):
        _, tff_name = str(args.dataset_name).split(":", 1)
    else:
        tff_name = str(args.dataset_name)
    tff_name = tff_name.lower()
    tag = make_load_tag(tff_name, benchmark="TFF")
    active_logger.info("[%s] loading federated client data.", tag)

    seq_len = int(args.seq_len)
    vocab_size = int(args.num_embeddings)

    if tff_name == "emnist":
        train_cd, _ = tff.simulation.datasets.emnist.load_data()
        client_ids = resolve_fixed_pool_clients(
            available_clients=list(train_cd.client_ids),
            args=args,
            prefix="tff",
        )
        if not client_ids:
            raise ValueError("No TFF clients selected for EMNIST.")
        active_logger.info("[%s] selected %d clients.", tag, len(client_ids))

        client_datasets = []
        for cid, client_id in enumerate(client_ids):
            tf_ds = train_cd.create_tf_dataset_for_client(client_id)
            xs, ys = [], []
            for ex in tf_ds:
                x = _tf_to_numpy(ex["pixels"]).astype(np.float32)
                y = int(_tf_to_numpy(ex["label"]))
                xs.append(torch.from_numpy(x).unsqueeze(0))
                ys.append(y)

            x_tensor = torch.stack(xs) if xs else torch.zeros(0, 1, 28, 28)
            y_tensor = torch.tensor(ys, dtype=torch.long)
            full_ds = BasicTensorDataset(
                x_tensor, y_tensor, name=f"[TFF-EMNIST] CLIENT<{cid}>"
            )

            indices = np.arange(len(full_ds))
            tr_ds, te_ds = split_subset_for_client(
                raw_train=full_ds,
                sample_indices=indices,
                client_id=cid,
                test_size=float(args.test_size),
                seed=int(args.seed),
            )
            client_datasets.append((tr_ds, te_ds))

        args = set_common_metadata(args, client_datasets)
        args.input_shape = (1, 28, 28)
        args.need_embedding = False
        args.seq_len = None
        args.num_embeddings = None
        active_logger.info("[%s] finished loading (%d clients).", tag, int(args.num_clients))
        return package_dataset_outputs(
            client_datasets=client_datasets,
            server_dataset=None,
            dataset_meta=args,
        )

    if tff_name == "celeba":
        train_cd, _ = tff.simulation.datasets.celeba.load_data()
        client_ids = resolve_fixed_pool_clients(
            available_clients=list(train_cd.client_ids),
            args=args,
            prefix="tff",
        )
        if not client_ids:
            raise ValueError("No TFF clients selected for CELEBA.")
        active_logger.info("[%s] selected %d clients.", tag, len(client_ids))

        client_datasets = []
        for cid, client_id in enumerate(client_ids):
            tf_ds = train_cd.create_tf_dataset_for_client(client_id)
            xs, ys = [], []
            for ex in tf_ds:
                x = _tf_to_numpy(ex["image"]).astype(np.float32)
                if x.ndim == 3:
                    x = np.transpose(x, (2, 0, 1))
                x = x / 255.0 if x.max() > 1 else x
                y_raw = _tf_to_numpy(ex["label"])
                if np.asarray(y_raw).ndim == 0:
                    y = int(y_raw)
                else:
                    y = int(np.asarray(y_raw).reshape(-1)[0])
                xs.append(torch.from_numpy(x))
                ys.append(y)

            x_tensor = torch.stack(xs) if xs else torch.zeros(0, 3, 84, 84)
            y_tensor = torch.tensor(ys, dtype=torch.long)
            full_ds = BasicTensorDataset(
                x_tensor, y_tensor, name=f"[TFF-CELEBA] CLIENT<{cid}>"
            )

            indices = np.arange(len(full_ds))
            tr_ds, te_ds = split_subset_for_client(
                raw_train=full_ds,
                sample_indices=indices,
                client_id=cid,
                test_size=float(args.test_size),
                seed=int(args.seed),
            )
            client_datasets.append((tr_ds, te_ds))

        args = set_common_metadata(args, client_datasets)
        args.need_embedding = False
        args.seq_len = None
        args.num_embeddings = None
        active_logger.info("[%s] finished loading (%d clients).", tag, int(args.num_clients))
        return package_dataset_outputs(
            client_datasets=client_datasets,
            server_dataset=None,
            dataset_meta=args,
        )

    if tff_name in {"shakespeare", "stackoverflow"}:
        if tff_name == "shakespeare":
            train_cd, _ = tff.simulation.datasets.shakespeare.load_data()
        else:
            train_cd, _ = tff.simulation.datasets.stackoverflow.load_data()

        client_ids = resolve_fixed_pool_clients(
            available_clients=list(train_cd.client_ids),
            args=args,
            prefix="tff",
        )
        if not client_ids:
            raise ValueError(f"No TFF clients selected for {tff_name}.")
        active_logger.info("[%s] selected %d clients.", tag, len(client_ids))
        client_datasets = []

        def encode_text(text: str):
            tokens = text.split() if tff_name == "stackoverflow" else list(text)
            ids = [_stable_hash_to_mod(str(tok), vocab_size) for tok in tokens]
            if len(ids) < seq_len:
                ids = ids + [0] * (seq_len - len(ids))
            return ids[:seq_len]

        for cid, client_id in enumerate(client_ids):
            tf_ds = train_cd.create_tf_dataset_for_client(client_id)
            xs, ys = [], []
            for ex in tf_ds:
                if tff_name == "shakespeare":
                    snippet = _tf_to_numpy(ex["snippets"])
                    next_c = _tf_to_numpy(ex["next_char"])
                    if isinstance(snippet, bytes):
                        snippet = snippet.decode("utf-8", errors="ignore")
                    if isinstance(next_c, bytes):
                        next_c = next_c.decode("utf-8", errors="ignore")
                    x_ids = encode_text(str(snippet))
                    y_id = _stable_hash_to_mod(str(next_c), vocab_size)
                else:
                    tokens = _tf_to_numpy(ex["tokens"])
                    if isinstance(tokens, bytes):
                        tokens = tokens.decode("utf-8", errors="ignore")
                    x_ids = encode_text(str(tokens))
                    y_id = x_ids[-1] if len(x_ids) > 0 else 0

                xs.append(torch.tensor(x_ids, dtype=torch.long))
                ys.append(int(y_id))

            x_tensor = (
                torch.stack(xs) if xs else torch.zeros(0, seq_len, dtype=torch.long)
            )
            y_tensor = torch.tensor(ys, dtype=torch.long)
            full_ds = BasicTensorDataset(
                x_tensor, y_tensor, name=f"[TFF-{tff_name.upper()}] CLIENT<{cid}>"
            )

            indices = np.arange(len(full_ds))
            tr_ds, te_ds = split_subset_for_client(
                raw_train=full_ds,
                sample_indices=indices,
                client_id=cid,
                test_size=float(args.test_size),
                seed=int(args.seed),
            )
            client_datasets.append((tr_ds, te_ds))

        args = set_common_metadata(args, client_datasets)
        train_chunks = [extract_targets(tr) for tr, _ in client_datasets if len(tr) > 0]
        if train_chunks:
            train_targets = np.concatenate(train_chunks).astype(np.int64)
            args.num_classes = int(train_targets.max()) + 1
        args.input_shape = (seq_len,)
        args.num_embeddings = vocab_size
        args.need_embedding = True
        args.seq_len = seq_len
        active_logger.info("[%s] finished loading (%d clients).", tag, int(args.num_clients))
        return package_dataset_outputs(
            client_datasets=client_datasets,
            server_dataset=None,
            dataset_meta=args,
        )

    raise NotImplementedError(
        f"Unsupported tff dataset '{tff_name}'. Supported: emnist, celeba, shakespeare, stackoverflow"
    )
