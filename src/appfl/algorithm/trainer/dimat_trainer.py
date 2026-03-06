"""
DIMATTrainer: Client-side trainer for the DIMAT algorithm.

Extends VanillaTrainer to reset BatchNorm running statistics after loading
the merged global model parameters, which is important for convergence
after DIMAT's feature-space alignment.
"""

import torch
import torch.nn as nn
from typing import Union, Dict, OrderedDict, Any
from appfl.algorithm.trainer.vanilla_trainer import VanillaTrainer


class DIMATTrainer(VanillaTrainer):
    """
    DIMATTrainer resets BatchNorm statistics after loading global parameters.
    After DIMAT merges models via activation matching, the BatchNorm running
    statistics need to be recomputed using the local training data.
    """

    def load_parameters(
        self,
        params: Union[Dict, OrderedDict, Any],
    ):
        """Load model parameters from the server.

        The server already resets BN stats using the full proxy dataset after
        merging, so we must NOT re-reset here with only local data — that would
        overwrite the server's higher-quality statistics and degrade accuracy.
        """
        super().load_parameters(params)

    def _reset_bn_stats(self):
        """Reset and recompute BatchNorm running statistics."""
        has_bn = False
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = None  # use simple average
                m.reset_running_stats()
                has_bn = True

        if not has_bn:
            return

        device = self.train_configs.get("device", "cpu")
        self.model.to(device)
        self.model.train()
        with torch.no_grad():
            for data in self.train_dataloader:
                if isinstance(data, (list, tuple)):
                    x = data[0].to(device)
                else:
                    x = data.to(device)
                self.model(x)
