from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from appfl.sim.datasets.common import (
    clientize_raw_dataset,
    finalize_dataset_outputs,
    infer_num_classes,
    make_load_tag,
    resolve_dataset_logger,
    to_namespace,
)


logger = logging.getLogger(__name__)


class GenericAudioDataset(Dataset):
    def __init__(self, base_dataset, fixed_num_frames: int = 16000):
        self.base_dataset = base_dataset
        self.fixed_num_frames = fixed_num_frames

        self.targets = []
        self.label_to_idx: Dict[str, int] = {}
        for i in range(len(self.base_dataset)):
            sample = self.base_dataset[i]
            label = self._extract_label(sample)
            key = str(label)
            if key not in self.label_to_idx:
                self.label_to_idx[key] = len(self.label_to_idx)
            self.targets.append(self.label_to_idx[key])
        self.targets = torch.tensor(self.targets, dtype=torch.long)

    @staticmethod
    def _extract_waveform(sample):
        if isinstance(sample, dict):
            for key in ["waveform", "audio", "input", "x"]:
                if key in sample:
                    return sample[key]
        if isinstance(sample, (tuple, list)):
            for value in sample:
                if torch.is_tensor(value) or isinstance(value, np.ndarray):
                    return value
        raise ValueError("Could not extract audio tensor from sample")

    @staticmethod
    def _extract_label(sample):
        if isinstance(sample, dict):
            for key in ["label", "target", "y", "class"]:
                if key in sample:
                    return sample[key]
        if isinstance(sample, (tuple, list)):
            # pick first scalar-like non tensor field after waveform
            for value in sample[1:]:
                if isinstance(value, (int, np.integer, str)):
                    return value
                if torch.is_tensor(value) and value.ndim == 0:
                    return int(value.item())
        return 0

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        sample = self.base_dataset[index]
        waveform = self._extract_waveform(sample)
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)
        elif not torch.is_tensor(waveform):
            waveform = torch.tensor(waveform)

        waveform = waveform.float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        if self.fixed_num_frames > 0:
            if waveform.shape[-1] < self.fixed_num_frames:
                pad = self.fixed_num_frames - waveform.shape[-1]
                waveform = torch.nn.functional.pad(waveform, (0, pad), value=0.0)
            elif waveform.shape[-1] > self.fixed_num_frames:
                waveform = waveform[..., : self.fixed_num_frames]

        target = self.targets[index]
        return waveform, target


def _resolve_torchaudio_dataset(name: str):
    try:
        import torchaudio
    except Exception:  # pragma: no cover
        return None, None

    ds_mod = torchaudio.datasets
    if hasattr(ds_mod, name):
        return getattr(ds_mod, name), torchaudio
    for candidate in dir(ds_mod):
        if candidate.lower() == name.lower():
            return getattr(ds_mod, candidate), torchaudio
    return None, torchaudio


def _instantiate_dataset(ds_cls, root: str, split: str, download: bool):
    sig = inspect.signature(ds_cls.__init__)
    kwargs = {}
    if "root" in sig.parameters:
        kwargs["root"] = root
    if "download" in sig.parameters:
        kwargs["download"] = download

    if "subset" in sig.parameters:
        subset_val = "training" if split == "train" else "testing"
        return ds_cls(subset=subset_val, **kwargs)

    if "train" in sig.parameters:
        kwargs["train"] = split == "train"
        return ds_cls(**kwargs)

    if "split" in sig.parameters:
        candidates = ["train", "training"] if split == "train" else ["test", "testing", "valid", "validation"]
        for cand in candidates:
            try:
                return ds_cls(split=cand, **kwargs)
            except Exception:
                continue

    return ds_cls(**kwargs)


def fetch_torchaudio_dataset(args):
    args = to_namespace(args)
    active_logger = resolve_dataset_logger(args, logger)
    tag = make_load_tag(str(args.dataset_name), benchmark="TORCHAUDIO")
    active_logger.info("[%s] resolving dataset class.", tag)
    ds_cls, ta = _resolve_torchaudio_dataset(args.dataset_name)
    if ta is None:
        raise RuntimeError("torchaudio is not installed.")
    if ds_cls is None:
        raise ValueError(f"Unknown torchaudio dataset: {args.dataset_name}")

    data_root = Path(str(args.data_dir)).expanduser()
    data_root.mkdir(parents=True, exist_ok=True)
    if bool(args.download):
        active_logger.info("[%s] downloading (if needed).", tag)

    train_base = _instantiate_dataset(
        ds_cls,
        str(data_root),
        "train",
        bool(args.download),
    )
    test_base = _instantiate_dataset(
        ds_cls,
        str(data_root),
        "test",
        bool(args.download),
    )

    max_frames = int(getattr(args, "audio_num_frames", 16000))
    raw_train = GenericAudioDataset(train_base, fixed_num_frames=max_frames)
    raw_test = GenericAudioDataset(test_base, fixed_num_frames=max_frames)
    active_logger.info("[%s] building federated client splits.", tag)

    client_datasets = clientize_raw_dataset(raw_train, args)
    client_datasets, server_dataset, dataset_meta = finalize_dataset_outputs(
        client_datasets=client_datasets,
        server_dataset=raw_test,
        dataset_meta=args,
        raw_train=raw_train,
    )
    dataset_meta.num_classes = int(infer_num_classes(raw_train))
    dataset_meta.need_embedding = False
    dataset_meta.seq_len = None
    dataset_meta.num_embeddings = None
    active_logger.info(
        "[%s] finished loading (%d clients).", tag, int(dataset_meta.num_clients)
    )
    return client_datasets, server_dataset, dataset_meta
