"""The ADKO v2 Suzuki surrogate: BoTorch's categorical GP on standardized yields.

Ported from the v2 reference implementation: a ``ScaleKernel`` over
``CategoricalKernel`` with one ARD lengthscale per dimension, refit from scratch after every
observation on yields standardized by their own mean and SD, predicted in batches of 1,000.
If a refit fails numerically, the previous model -- and its standardization -- is kept, which
is the one place this differs from the v1 example's surrogate.

``predict`` returns standardized ``(mu, sigma)``, which is what the v2 score adds its
``match_rms``-calibrated peer terms to; ``raw_mean`` maps a mean back to yield, which is what
trust encodes a prediction from.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Sequence, Tuple

import gpytorch
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

from appfl.decentralized.algorithm.adko_v2 import Surrogate


class CategoricalSingleTaskGP(ExactGP, GPyTorchModel):
    """The reference's GP (Olympus' BoTorch wrapper): constant mean, scaled categorical kernel."""

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


class CategoricalGP(Surrogate):
    """One lab's private GP over its own ``(phi(theta), y)`` pairs."""

    #: Posterior batch size; keeps full-grid prediction memory bounded.
    BATCH_SIZE = 1000

    def __init__(self) -> None:
        self._X: List[List[float]] = []
        self._y: List[float] = []
        self.model: Optional[CategoricalSingleTaskGP] = None
        self.mean, self.std = 0.0, 1.0

    @property
    def fitted(self) -> bool:
        return self.model is not None

    def update(self, embedding: Sequence[float], observation: float) -> None:
        """Append to ``D_i`` and refit from scratch."""
        self._X.append([float(v) for v in embedding])
        self._y.append(float(observation))
        self._fit()

    def _fit(self) -> None:
        y = np.asarray(self._y, dtype=np.float64)
        if len(y) < 2:
            return
        mean, std = float(y.mean()), float(y.std()) if y.std() > 0 else 1.0
        y_std = (y - mean) / std
        Xt = torch.from_numpy(np.asarray(self._X, dtype=np.float64))
        yt = torch.from_numpy(y_std.reshape(-1, 1))
        model = CategoricalSingleTaskGP(Xt, yt).double()
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fit_gpytorch_mll(mll)
            except Exception:
                return  # keep the previous model if this fit is numerically unstable
        self.model, self.mean, self.std = model, mean, std

    def predict(self, X, batch: int = BATCH_SIZE) -> Tuple[np.ndarray, np.ndarray]:
        """Standardized posterior mean and SD; ``(0, 1)`` before any fit."""
        X = np.asarray(X, dtype=np.float64)
        if self.model is None:
            return np.zeros(len(X)), np.ones(len(X))
        self.model.eval()
        mus, sds = [], []
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(0, len(X), batch):
                post = self.model.posterior(torch.from_numpy(X[i : i + batch]))
                mus.append(post.mean.squeeze(-1).numpy())
                var = post.variance.squeeze(-1).numpy()
                sds.append(np.sqrt(np.maximum(var, 0.0)))
        return np.concatenate(mus), np.concatenate(sds)

    def raw_mean(self, mu) -> np.ndarray:
        return np.asarray(mu) * self.std + self.mean


def get_surrogate() -> CategoricalGP:
    """Entry point named by ``surrogate_configs.surrogate_name``. Every knob is the kernel's."""
    return CategoricalGP()
