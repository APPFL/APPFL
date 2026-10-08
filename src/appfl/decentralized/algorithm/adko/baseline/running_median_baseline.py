"""Running-median ADKO baseline.

Rillo et al. (ADKO) recommend a local-history baseline when there is no natural
success threshold. This uses ``b = median(y)`` and a median-absolute-deviation
scale, so one unusually good or bad observation does not dominate future token
confidence.
"""

from __future__ import annotations

from typing import List, Tuple

from appfl.decentralized.algorithm.adko.baseline.base_baseline import BaseBaseline


def median(values: List[float]) -> float:
    """Return the median value."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    return ordered[mid] if n % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])


def standard_deviation(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


class RunningMedianBaseline(BaseBaseline):
    """Use ``b = median(y)`` and ``scale = 1.4826 * median(|y - b|)``.

    Args:
        warmup_baseline: Initial ``b`` before observations.
        warmup_scale: Initial ``scale`` before observations.
    """

    MAD_TO_SIGMA = 1.4826

    def __init__(self, warmup_baseline: float = 0.0, warmup_scale: float = 1.0):
        self.warmup_baseline = float(warmup_baseline)
        self.warmup_scale = float(warmup_scale)
        self._history: List[float] = []

    def observe(self, observation: float) -> None:
        self._history.append(float(observation))

    def current(self) -> Tuple[float, float]:
        if not self._history:
            return self.warmup_baseline, self.warmup_scale
        baseline = median(self._history)
        scale = self.MAD_TO_SIGMA * median([abs(y - baseline) for y in self._history])
        if scale < 1e-8:
            scale = standard_deviation(self._history)
        if scale < 1e-8:
            scale = 1.0
        return baseline, scale

    @property
    def n_observations(self) -> int:
        return len(self._history)
