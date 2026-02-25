from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from appfl.sim.algorithm.trainer.base_trainer import BaseTrainer


class PrivacyFutureTrainer(BaseTrainer):
    """Reserved trainer slot for future DP/Secure Aggregation features."""

    def __init__(
        self,
        *args,
        train_configs: DictConfig | None = None,
        **kwargs,
    ) -> None:
        del args, kwargs, train_configs
        raise NotImplementedError(
            "PrivacyFutureTrainer is a placeholder. DP/Secure Aggregation paths are intentionally excluded from FedavgTrainer."
        )

    def train(self, **kwargs) -> dict[str, Any]:
        del kwargs
        raise NotImplementedError("PrivacyFutureTrainer is not implemented yet.")
