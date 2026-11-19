"""Private GP surrogate for the CIFAR-10 study -- the reference's ``GPSurrogate``.

A scikit-learn GP with an ARD Matern-5/2 kernel (one length scale per design dimension),
``alpha`` jitter, ``n_restarts`` optimizer restarts and ``normalize_y=True``, refit after every
observation. Mirrors ``cifar10/adko.py::GPSurrogate`` in github.com/lucasrillo/adko.

One deliberate difference from ``categorical_gp_surrogate.py``: ``posterior`` returns ``mu`` and
``sigma`` on the *accuracy* scale, not standardized. With ``normalize_y=True`` scikit-learn
un-normalizes its predictions, and that is what the reference's reasoning score consumes -- its
``lam`` and ``gamma`` were chosen against accuracy-scale ``mu`` and ``sigma``. Standardizing here
would quietly change how much the peer terms count.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from appfl.decentralized.algorithm.adko import Surrogate


class MaternGPSurrogate(Surrogate):
    """One agent's private GP over ``(phi(theta), validation accuracy)`` pairs."""

    def __init__(self, dim: int = 7, alpha: float = 0.01, n_restarts: int = 2):
        kernel = Matern(
            nu=2.5, length_scale=np.ones(dim), length_scale_bounds=(1e-3, 1e3)
        )
        self._gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            n_restarts_optimizer=n_restarts,
            normalize_y=True,
        )
        self._X: list[list[float]] = []
        self._y: list[float] = []
        self._fitted = False

    def posterior(
        self, candidates: Sequence[Sequence[float]]
    ) -> list[tuple[float, float]]:
        n = len(candidates)
        if not self._fitted or n == 0:
            return [(0.0, 1.0)] * n  # the reference's prior before two observations
        mu, sigma = self._gp.predict(
            np.asarray(candidates, dtype=float), return_std=True
        )
        return list(zip(mu.tolist(), sigma.tolist()))

    def update(self, embedding: Sequence[float], observation: float) -> None:
        """Algorithm 1 step 12: append to ``D_i`` and refit."""
        self._X.append([float(v) for v in embedding])
        self._y.append(float(observation))
        if len(self._y) < 2:
            return
        # "Optimal value close to bound" is informational and arrives on most refits.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self._gp.fit(np.asarray(self._X), np.asarray(self._y))
        self._fitted = True


def get_surrogate(
    dim: int = 7, alpha: float = 0.01, n_restarts: int = 2
) -> MaternGPSurrogate:
    """Entry point named by ``surrogate_configs.surrogate_name``."""
    return MaternGPSurrogate(dim=dim, alpha=alpha, n_restarts=n_restarts)
