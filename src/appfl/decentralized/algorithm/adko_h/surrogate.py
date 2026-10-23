"""A small exact GP, because every arm must share one surrogate.

ADKO v2 gets its surrogate from the application (Suzuki's categorical GP lives in APPFL's
``examples/``). The arms ADKO-H compares differ in planning, not in modelling, so the
comparison is only meaningful if every arm -- plain ADKO, the diversity-aware baseline, the
random-campaign control and the LLM -- fits the identical model. Hence one implementation
here, deliberately plain: zero mean, RBF kernel, fixed noise, a median-heuristic lengthscale,
and a Cholesky solve.

Two consequences worth stating rather than hiding:

* a GP over a few hundred points with a heuristic lengthscale is weaker than the tuned
  surrogates on a benchmark leaderboard, so absolute numbers here sit below published ones;
  every arm pays that equally, which is what the claim rests on;
* ``max_points`` caps the fit at the most recent points, so cost stays linear in the rounds
  rather than cubic in the whole run -- what matters at a thousand agents.

``predict`` returns **standardized** mean and SD, as ``adko_v2.components.Surrogate``
requires: the v2 score adds them to a social term calibrated against ``sigma``, so the two
have to be in the same units.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from appfl.decentralized.algorithm.adko_v2.components import Surrogate


class GP(Surrogate):
    """Zero-mean RBF GP on z-scored outcomes.

    Args:
        noise: observation noise on the standardized scale.
        lengthscale: fixed lengthscale, or ``None`` for the median heuristic.
        max_points: most recent points to fit; older ones are dropped from the fit only.
        jitter: added to the diagonal if the Cholesky fails.
    """

    def __init__(
        self,
        noise: float = 1e-2,
        lengthscale: Optional[float] = None,
        max_points: int = 600,
        jitter: float = 1e-8,
    ):
        self.noise = float(noise)
        self.fixed_ls = lengthscale
        self.max_points = int(max_points)
        self.jitter = float(jitter)
        self._X: List[np.ndarray] = []
        self._y: List[float] = []
        self._dirty = True
        self._ls = 1.0
        self._mean = 0.0
        self._sd = 1.0
        self._L: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None
        self._Xf: Optional[np.ndarray] = None

    @property
    def fitted(self) -> bool:
        return len(self._y) >= 2

    def update(self, embedding: Sequence[float], observation: float) -> None:
        self._X.append(np.asarray(embedding, dtype=float).ravel())
        self._y.append(float(observation))
        self._dirty = True

    def _fit(self) -> None:
        X = np.asarray(self._X[-self.max_points :], dtype=float)
        y = np.asarray(self._y[-self.max_points :], dtype=float)
        self._mean = float(np.mean(y))
        self._sd = float(np.std(y)) or 1.0
        z = (y - self._mean) / self._sd
        self._ls = float(self.fixed_ls) if self.fixed_ls else _median_lengthscale(X)
        K = _rbf(X, X, self._ls) + self.noise * np.eye(len(X))
        for extra in (0.0, self.jitter, 1e-6, 1e-4):
            try:
                self._L = np.linalg.cholesky(K + extra * np.eye(len(X)))
                break
            except np.linalg.LinAlgError:
                self._L = None
        if self._L is None:
            self._alpha, self._Xf = None, None
            return
        self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, z))
        self._Xf = X
        self._dirty = False

    def predict(self, X: Sequence[Sequence[float]]) -> Tuple[np.ndarray, np.ndarray]:
        """Standardized posterior mean and SD; ``(0, 1)`` before a fit, as v2 specifies."""
        Xs = np.atleast_2d(np.asarray(X, dtype=float))
        if not self.fitted:
            return np.zeros(len(Xs)), np.ones(len(Xs))
        if self._dirty:
            self._fit()
        if self._alpha is None or self._Xf is None:
            return np.zeros(len(Xs)), np.ones(len(Xs))
        Ks = _rbf(Xs, self._Xf, self._ls)
        mu = Ks @ self._alpha
        v = np.linalg.solve(self._L, Ks.T)
        var = np.maximum(1.0 - np.sum(v * v, axis=0), 1e-12)
        return mu, np.sqrt(var)

    def raw_mean(self, mu: Sequence[float]) -> np.ndarray:
        return np.asarray(mu, dtype=float) * self._sd + self._mean


def _rbf(A: np.ndarray, B: np.ndarray, ls: float) -> np.ndarray:
    d2 = np.sum(A * A, axis=1)[:, None] + np.sum(B * B, axis=1)[None, :] - 2.0 * A @ B.T
    return np.exp(-0.5 * np.maximum(d2, 0.0) / (ls * ls))


def _median_lengthscale(X: np.ndarray) -> float:
    """Median pairwise distance, on a subsample when the history is long."""
    if len(X) < 3:
        return 1.0
    idx = np.arange(len(X)) if len(X) <= 200 else np.linspace(0, len(X) - 1, 200).astype(int)
    S = X[idx]
    d2 = np.sum(S * S, axis=1)[:, None] + np.sum(S * S, axis=1)[None, :] - 2.0 * S @ S.T
    d = np.sqrt(np.maximum(d2, 0.0))
    med = float(np.median(d[np.triu_indices(len(S), 1)])) if len(S) > 1 else 1.0
    return max(med, 1e-3)
