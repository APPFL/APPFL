"""
Placeholder model for the Evo2 PEFT server config.

The server never actually trains or holds a model — clients train Evo2
7B + LoRA/QLoRA/IA3 adapters locally and only ever exchange adapter tensors.
`client_configs.model_configs` still requires some importable `nn.Module`
class, though, so this is it: a module with no parameters, whose empty
`state_dict()` is exactly what `Evo2FedAvgAggregator.get_parameters()`
checks for to decide it should fall back to the aggregated adapter state
instead of reading from `self.model`.
"""

import torch.nn as nn


class DummyModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        return x
