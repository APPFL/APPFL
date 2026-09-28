"""Dependency-free surrogate for the toy experiment -- the *model* an agent fits privately.

The decentralized analogue of ``examples/resources/model``: named by path in the federation
config, instantiated once per agent, and never transmitted (ADKO Constraint 3.1).
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from appfl.decentralized.algorithm.adko import Surrogate


class KernelSurrogate(Surrogate):
    """Nadaraya-Watson mean with distance-based uncertainty. Stands in for a GP.

    Dependency-free on purpose, so the demos run anywhere APPFL runs. It reproduces the two
    properties the reasoning score depends on: a mean that tracks observations, and a sigma
    that grows away from them so ``beta * sigma`` drives exploration.
    """

    def __init__(self, bandwidth: float = 0.05, prior_sigma: float = 1.0):
        self.bandwidth = bandwidth
        self.prior_sigma = prior_sigma
        self.xs: List[float] = []
        self.ys: List[float] = []

    def posterior(self, candidates: Sequence[Sequence[float]]) -> List[Tuple[float, float]]:
        """Return **standardized** ``(mu, sigma)``.

        Standardizing here is not cosmetic. The peer terms G and Lambda are normalized into
        [0, 1] by construction, so a posterior left on the raw 0-100 yield scale would swamp
        them no matter how lam and gamma are set. The reference standardizes for the same
        reason (``mu_std``, ``sigma_std`` in ``run_suzuki.py``), and the surrogate is the
        right place for it -- it is the only component that knows the objective's scale.
        """
        out = []
        spread = self._spread()
        center = sum(self.ys) / len(self.ys) if self.ys else 0.0
        for candidate in candidates:
            x = candidate[0]
            if not self.xs:
                out.append((0.0, self.prior_sigma))
                continue
            weights = [
                math.exp(-((x - xi) ** 2) / (2 * self.bandwidth**2)) for xi in self.xs
            ]
            total = sum(weights)
            mean = (
                sum(w * y for w, y in zip(weights, self.ys)) / total
                if total > 1e-12
                else 0.0
            )
            nearest = min(abs(x - xi) for xi in self.xs)
            sigma = self.prior_sigma * (
                1.0 - math.exp(-((nearest / self.bandwidth) ** 2))
            )
            out.append(((mean - center) / spread, sigma))
        return out

    def _spread(self) -> float:
        if len(self.ys) < 2:
            return 1.0
        center = sum(self.ys) / len(self.ys)
        var = sum((y - center) ** 2 for y in self.ys) / len(self.ys)
        return max(math.sqrt(var), 1e-8)

    def update(self, embedding: Sequence[float], observation: float) -> None:
        self.xs.append(embedding[0])
        self.ys.append(observation)


def get_surrogate(bandwidth: float = 0.05, prior_sigma: float = 1.0) -> KernelSurrogate:
    """Entry point named by ``surrogate_configs.surrogate_name``."""
    return KernelSurrogate(bandwidth=bandwidth, prior_sigma=prior_sigma)
