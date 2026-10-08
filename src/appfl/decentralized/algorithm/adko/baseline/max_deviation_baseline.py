"""Max-deviation baseline: b from a chosen rule; scale is max |y - b|.

Implements the paper's ||y-b||_max using only local data.

Two common variants:

- Fixed-threshold: b = tau (domain constant); scale = max_k |y_k - tau| over local history.
- Running-median: b = median(y_history); scale = max_k |y_k - b|, where b is the current
  median. This mirrors the many-task intent but replaces MAD with a max deviation.

`mode` controls how the max is computed:

- "per_obs" (default): update a running maximum at observe-time using the current b.
- "current": recompute max(|y - b_current|) over the whole history each time `current()` is
  called. This may shrink if b moves; a `min_scale` floor prevents degeneracy.

Scale is always lower-bounded by `min_scale` so `c = |y - b| / scale` remains well-defined
in early rounds or flat histories.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from appfl.decentralized.algorithm.adko.baseline.base_baseline import BaseBaseline
from appfl.decentralized.algorithm.adko.baseline.running_median_baseline import median


class MaxDeviationBaseline(BaseBaseline):
    def __init__(
        self,
        fixed_threshold: Optional[float] = None,
        warmup_baseline: float = 0.0,
        warmup_scale: float = 1.0,
        mode: str = "per_obs",  # "per_obs" | "current"
        min_scale: float = 1.0,
    ):
        if mode not in ("per_obs", "current"):
            raise ValueError("mode must be 'per_obs' or 'current'")
        self.fixed_threshold = fixed_threshold
        self.warmup_baseline = float(warmup_baseline)
        self.warmup_scale = float(warmup_scale)
        self.mode = mode
        self.min_scale = float(min_scale)
        self._history: List[float] = []
        self._running_max: float = 0.0

    def _baseline(self) -> float:
        if self.fixed_threshold is not None:
            return float(self.fixed_threshold)
        if not self._history:
            return self.warmup_baseline
        return median(self._history)

    def observe(self, observation: float) -> None:
        """
        Updates local history and for mode 'per_obs', updates a running maximum of |y-b| using the current b. The logic is as follows:
        1. Append y to history
        2. Compute the current baseline b (either fixed tau or running median of history)
        3. dev = |y - b|
        4. _running_max = max(_running_max, dev)
        """
        self._history.append(float(observation))
        if self.mode == "per_obs":
            b = self._baseline()
            dev = abs(observation - b)
            if dev > self._running_max:
                self._running_max = dev

    def current(self) -> Tuple[float, float]:
        """
        Returns (b, scale) used for the next token encoding. There are two modes:
        1. per_obs (default): scale = max observed |y-b| (running max) with a min_scale floor
        2. current: recompute scale = max(|y-b_current|) over the full history with the same floor
        """
        if not self._history:
            return self.warmup_baseline, max(self.warmup_scale, self.min_scale)
        b = self._baseline()
        if self.mode == "per_obs":
            scale = max(self._running_max, self.min_scale)
        else:  # "current"
            scale = max(max(abs(y - b) for y in self._history), self.min_scale)
        return b, scale
