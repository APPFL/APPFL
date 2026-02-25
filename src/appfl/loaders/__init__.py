"""
appfl.loaders: unified dataset loading for APPFL simulations.

Requires appfl[sim] to be installed.

Supported backends: torchvision, torchtext, torchaudio, medmnist,
huggingface (hf), flamby, leaf, tff, custom, auto.

Usage::

    from appfl.loaders import load_dataset
    client_datasets, server_dataset, dataset_meta = load_dataset(cfg)
"""

from appfl.sim.loaders.data import load_dataset  # noqa: F401

__all__ = ["load_dataset"]
