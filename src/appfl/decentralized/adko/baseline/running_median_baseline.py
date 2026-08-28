"""Median of local history, scaled by median absolute deviation."""

from __future__ import annotations

from typing import List, Tuple

from appfl.decentralized.adko.baseline.base_baseline import BaseBaseline


def median(values: List[float]) -> float:
    """Plain median. Kept module-level so custom baselines can reuse it."""
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
    """``b = median(y)``, ``scale = 1.4826 * median(|y - b|)``. The paper's v2 default.

    The constant rescales median absolute deviation into an estimate of a Gaussian standard
    deviation. Both statistics are robust, which is the point: an agent that hits one
    spectacular outlier early does not spend the rest of the run calling everything else a
    failure -- the failure mode a mean-and-standard-deviation version has.

    Falls back to the standard deviation when MAD degenerates (more than half the history
    identical), and to 1.0 when that degenerates too, so a flat early history cannot produce a
    divide-by-zero or a token whose advantage is meaningless.
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
