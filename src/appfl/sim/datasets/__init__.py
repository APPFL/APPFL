from appfl.sim.datasets.flambyparser import fetch_flamby
from appfl.sim.datasets.customparser import fetch_custom_dataset
from appfl.sim.datasets.hfparser import fetch_hf_dataset
from appfl.sim.datasets.leafparser import fetch_leaf
from appfl.sim.datasets.medmnistparser import fetch_medmnist_dataset
from appfl.sim.datasets.tffparser import fetch_tff_dataset
from appfl.sim.datasets.torchaudioparser import fetch_torchaudio_dataset
from appfl.sim.datasets.torchtextparser import fetch_torchtext_dataset
from appfl.sim.datasets.torchvisionparser import fetch_torchvision_dataset

__all__ = [
    "fetch_custom_dataset",
    "fetch_hf_dataset",
    "fetch_flamby",
    "fetch_leaf",
    "fetch_torchvision_dataset",
    "fetch_torchtext_dataset",
    "fetch_torchaudio_dataset",
    "fetch_medmnist_dataset",
    "fetch_tff_dataset",
]
