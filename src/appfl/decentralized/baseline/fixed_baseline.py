"""A constant, federation-wide threshold."""

from __future__ import annotations

from typing import Tuple

from appfl.decentralized.baseline.base_baseline import BaseBaseline


class FixedBaseline(BaseBaseline):
    """A constant threshold shared by every agent. The Suzuki configuration.

    ``scale`` normalizes the advantage score. The reference derives it from the objective's
    known range as ``max(tau, 100 - tau)``, which for ``tau = 50`` gives 50.

    Because it is identical across agents, a SUCCESS from any agent means the same thing --
    which is what makes summing peer evidence coherent. That property is exactly what a
    per-agent running baseline gives up, and why the choice is domain-dependent rather than a
    matter of taste.
    """

    def __init__(self, threshold: float, scale: float):
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        self.threshold = float(threshold)
        self.scale = float(scale)

    def observe(self, observation: float) -> None:
        """No-op: a fixed baseline does not learn from local history."""
        return None

    def current(self) -> Tuple[float, float]:
        return self.threshold, self.scale
