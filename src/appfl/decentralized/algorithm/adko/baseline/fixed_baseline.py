"""Fixed ADKO baseline.

Use this when the experiment has a natural global success threshold, such as the
Suzuki benchmark yield cutoff in Rillo et al. (ADKO). Every agent uses the same
``b`` and ``scale``, so a success token has the same meaning across the
federation.
"""

from __future__ import annotations

from typing import Tuple

from appfl.decentralized.algorithm.adko.baseline.base_baseline import BaseBaseline


class FixedBaseline(BaseBaseline):
    """Constant ``b = threshold`` and ``scale``.

    Args:
        threshold: Fixed baseline ``b``.
        scale: Divides ``|y - b|`` to get confidence ``c``.
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
