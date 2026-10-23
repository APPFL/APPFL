"""Private per-neighbour trust: product agreement, Bush-Mosteller updates, odds weights.

On by default in the settled method (``ADKO-v2f-odds-bit``). Each agent keeps ``t_j`` in
``[0, 1]`` for every neighbour ``j``, starting at the neutral ``1/2``, and never transmits it.

Agreement. When a token ``k`` from ``j`` arrives, I encode my own GP's prediction at its input
exactly as I would encode a result -- ``q_hat`` against *my* median, scale and codebook -- and
score how far the two agree, shrunk toward ``1/2`` where I am uncertain::

    a_k = 1/2 + ((1 + q_k * q_hat_k)/2 - 1/2) * (1 - sigma(x_k) / max_Z sigma)

Update. Once a round per neighbour, with the mean agreement ``a`` of its new tokens and
``eta = 0.1``::

    t <- t + eta * ([2a - 1]_+ (1 - t) - [1 - 2a]_+ t)

Weights. Sources are mixed by the odds ``t / (1 - t)``: 1 at neutral, never 0, so an unrelated
peer is down-weighted rather than cut off. The findings measured odds against a floored gate
and against no trust: odds gains where tasks diverge (3 and 6 groups, the widest spreads) and
ties elsewhere, which is why it is on by default.

Trust learns only after warmup and once my GP exists. Ported from the reference implementation.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np


def agreement(q_tok: float, q_hat: float, sigma_tilde: float) -> float:
    """``(1 + q * q_hat) / 2``, shrunk toward 1/2 by my relative uncertainty."""
    raw = 0.5 * (1.0 + float(q_tok) * float(q_hat))
    return 0.5 + (raw - 0.5) * (1.0 - float(np.clip(sigma_tilde, 0, 1)))


def relative_sigma(sigma_x, sigma_ref: Optional[float]) -> np.ndarray:
    """``sigma(x) / max_Z sigma`` in ``[0, 1]``; 1 (uninformed) when undefined."""
    sigma_x = np.asarray(sigma_x, dtype=float)
    if not sigma_ref or sigma_ref <= 0:
        return np.ones_like(sigma_x)
    return np.clip(sigma_x / float(sigma_ref), 0.0, 1.0)


def _odds(t: float) -> float:
    t = min(max(float(t), 1e-6), 1.0 - 1e-6)
    return t / (1.0 - t)


class TrustState:
    """``t_j`` per neighbour, 1/2 neutral. Never transmitted.

    :param neighbors: neighbour indices known up front. A deployed site may not know its
        peers' indices before their first token, so :meth:`ensure` adds a source when it is
        first heard from -- at the same neutral 1/2, which is what an up-front entry would have
        held, so the two are indistinguishable.
    """

    def __init__(
        self, neighbors: Iterable[int] = (), eta: float = 0.1, init: float = 0.5
    ):
        if not 0.0 < eta <= 1.0:
            raise ValueError("eta must be in (0, 1]")
        self.eta, self.init = float(eta), float(init)
        self.t: Dict[int, float] = {int(j): self.init for j in neighbors}

    def ensure(self, j: int) -> None:
        """Start tracking ``j`` at the neutral value if it is not tracked yet."""
        self.t.setdefault(int(j), self.init)

    def update(self, j: int, agreements: Sequence[float]) -> float:
        """One Bush-Mosteller step with the round's mean agreement for ``j``."""
        a = np.asarray(agreements, dtype=float)
        j = int(j)
        self.ensure(j)
        if a.size == 0:
            return self.t[j]
        a = float(np.clip(a.mean(), 0.0, 1.0))
        t = self.t[j]
        up, down = max(2 * a - 1, 0.0), max(1 - 2 * a, 0.0)
        self.t[j] = min(max(t + self.eta * (up * (1 - t) - down * t), 0.0), 1.0)
        return self.t[j]

    def weights(self) -> Dict[int, float]:
        """Source weights: the odds ``t / (1 - t)`` of each tracked neighbour."""
        return {j: _odds(t) for j, t in self.t.items()}

    def mean(self) -> float:
        """Mean trust over tracked neighbours; 1/2 when there are none."""
        return float(np.mean(list(self.t.values()))) if self.t else self.init

    def snapshot(self) -> Dict[int, float]:
        return dict(sorted(self.t.items()))
