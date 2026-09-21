"""Reference-compatible private GP surrogate for Suzuki.

The surrogate is the model each ADKO agent fits to its own measured yields. It
predicts ``mu`` and ``sigma`` for candidate reaction conditions before the agent
spends a real evaluation on one of them.

This file mirrors the reference Suzuki GP: categorical inputs, a
``CategoricalKernel``, standardized yields, one refit after each new
observation, and chunked posterior prediction. ``posterior`` deliberately
returns standardized ``mu`` and ``sigma`` so the GP terms stay on the same scale
as the normalized peer terms ``G`` and ``Lambda``.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.kernels.categorical import CategoricalKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP

from appfl.decentralized.algorithm.adko import Surrogate


class CategoricalSingleTaskGP(ExactGP, GPyTorchModel):
    """BoTorch GP used by the reference Suzuki implementation.

    Inputs are integer category IDs. The covariance is a scaled categorical
    kernel, so similarity is learned over category matches rather than numeric
    distance between IDs.
    """

    _num_outputs = 1

    def __init__(self, train_X: torch.Tensor, train_Y: torch.Tensor):
        super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            base_kernel=CategoricalKernel(ard_num_dims=train_X.shape[-1])
        )
        self.to(train_X)

    def forward(self, x):  # noqa: D102 -- gpytorch interface
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


class CategoricalGPSurrogate(Surrogate):
    """Private Suzuki surrogate used by one ADKO agent.

    The model stores that agent's local ``(phi(theta), y)`` observations. Each
    update appends one observation and refits the GP from scratch, matching the
    reference behavior.
    """

    #: Reference posterior batch size. Keeps full-grid prediction memory bounded.
    BATCH_SIZE = 1000

    def __init__(self) -> None:
        self._X: List[List[float]] = []
        self._y: List[float] = []
        self._gp: Optional[CategoricalSingleTaskGP] = None
        self._y_mean = 0.0
        self._y_std = 1.0

    def posterior(
        self, candidates: Sequence[Sequence[float]]
    ) -> List[Tuple[float, float]]:
        n = len(candidates)
        # With fewer than two observations, use the reference's flat prior.
        if self._gp is None or len(self._y) < 2 or n == 0:
            return [(0.0, 1.0)] * n

        import gpytorch

        Xc = torch.from_numpy(np.asarray(candidates, dtype=np.float64))
        self._gp.eval()
        mus: List[np.ndarray] = []
        sigmas: List[np.ndarray] = []
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(0, Xc.shape[0], self.BATCH_SIZE):
                posterior = self._gp.posterior(Xc[i : i + self.BATCH_SIZE])
                mu = posterior.mean.squeeze(-1).cpu().numpy()
                var = posterior.variance.squeeze(-1).cpu().numpy()
                mus.append(mu)
                sigmas.append(np.sqrt(np.maximum(var, 0.0)))
        mu_all = np.concatenate(mus)
        sigma_all = np.concatenate(sigmas)
        return list(zip(mu_all.tolist(), sigma_all.tolist()))

    def posterior_arrays(
        self, candidates: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return posterior arrays for the full-grid scoring path."""
        n = len(candidates)
        if self._gp is None or len(self._y) < 2 or n == 0:
            return np.zeros(n), np.ones(n)

        import gpytorch

        Xc = torch.from_numpy(np.asarray(candidates, dtype=np.float64))
        self._gp.eval()
        mus: List[np.ndarray] = []
        sigmas: List[np.ndarray] = []
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(0, Xc.shape[0], self.BATCH_SIZE):
                posterior = self._gp.posterior(Xc[i : i + self.BATCH_SIZE])
                mus.append(posterior.mean.squeeze(-1).cpu().numpy())
                var = posterior.variance.squeeze(-1).cpu().numpy()
                sigmas.append(np.sqrt(np.maximum(var, 0.0)))
        return np.concatenate(mus), np.concatenate(sigmas)

    def update(self, embedding: Sequence[float], observation: float) -> None:
        """Algorithm 1 step 12: append to ``D_i`` and refit."""
        self._X.append([float(v) for v in embedding])
        self._y.append(float(observation))
        self._fit()

    # -- internals ---------------------------------------------------------------------

    def _fit(self) -> None:
        if len(self._y) < 2:
            return
        X = torch.from_numpy(np.asarray(self._X, dtype=np.float64))
        y = np.asarray(self._y, dtype=np.float64)
        # Fit on standardized yields so GP terms stay comparable to peer terms.
        self._y_mean = float(y.mean())
        self._y_std = float(y.std()) if y.std() > 0 else 1.0
        y_std_t = torch.from_numpy(
            ((y - self._y_mean) / self._y_std).reshape(-1, 1)
        )
        gp = CategoricalSingleTaskGP(X, y_std_t).double()
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fit_gpytorch_mll(mll)
                self._gp = gp
            except Exception:
                # Keep the previous GP if this round's fit is numerically unstable.
                pass
