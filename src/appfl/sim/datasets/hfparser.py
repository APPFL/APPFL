from __future__ import annotations

from appfl.sim.datasets.common import to_namespace


def _resolve_hf_dataset_name(args) -> str:
    raw_dataset = str(getattr(args, "dataset_name", "")).strip()
    explicit_name = str(getattr(args, "ext_dataset_name", "")).strip()

    if raw_dataset.lower().startswith("hf:"):
        name = raw_dataset.split(":", 1)[1].strip()
    elif explicit_name:
        name = explicit_name
    elif raw_dataset and ":" not in raw_dataset:
        name = raw_dataset
    else:
        raise ValueError(
            "Unable to infer HuggingFace dataset name. "
            "Use dataset.name='hf:<repo>' or set dataset.name='<repo>' with dataset.backend=hf."
        )
    if not name:
        raise ValueError("HuggingFace dataset name is empty.")
    return name


def fetch_hf_dataset(args):
    args = to_namespace(args)
    dataset_name = _resolve_hf_dataset_name(args)
    args.ext_source = "hf"
    args.ext_dataset_name = dataset_name

    # Reuse HF loading helper implementation.
    from appfl.sim.datasets.externalparser import _fetch_hf_dataset

    return _fetch_hf_dataset(args, dataset_name)
