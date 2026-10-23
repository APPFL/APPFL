"""The ADKO v2 reasoning score -- what each agent argmaxes to pick its next probe.

    R_i(x) = mu_i(x)  +  beta * sigma_i(x)  +  lam * G_i(x)  -  gamma * Lambda_i(x)

with ``mu`` and ``sigma`` the agent's own GP mean and SD, both **standardized** (the GP is fit
to z-scored outcomes), and ``beta = lam = gamma = 2`` everywhere in the settled method. The
first two terms are GP-UCB over private data; the last two are the collaboration, computed
from peer tokens only.

The peer field. Each source ``j`` with tokens in my memory contributes the average of its
tokens' kernel-weighted scores, split by sign, and sources are mixed by weights ``w_j`` that
sum to one over the sources present::

    G0(x) = sum_j w_j (1/|K_j|) sum_{k in K_j} [q_k]_+ S(x, x_k)
    P0(x) = sum_j w_j (1/|K_j|) sum_{k in K_j} [-q_k]_+ S(x, x_k)       0 <= G0 + P0 <= 1

``w_j`` is uniform, or proportional to the odds ``t_j / (1 - t_j)`` of my trust in ``j`` when
trust is on (see ``trust.py``). Averaging inside a source gives every neighbour one voice
however many of its tokens survived the FIFO.

The kernel is ``exp(-d^2 / sigma^2)`` for Euclidean inputs and ``exp(-h^2 / 2 sigma^2)`` for
categorical ones under Hamming distance ``h``, with ``sigma = 1`` and ``0.5`` respectively.

Calibration, ``match_rms``. The raw field lives on ``[0, 1]`` while ``mu`` and ``sigma`` are in
standardized units, so once a round, on the reference set ``Z`` (the agent's feasible grid),
one factor ``k`` is fixed so that the social term has the same RMS as the exploration term::

    k = RMS_Z(sigma) / max(RMS_Z(G0 + P0), eps)        G = k G0,   Lambda = k P0

The factor is frozen for the round. This is the reference's ``match_rms``: the findings show
that gating the field by ``sigma`` (v2's first calibration) erased the collaborative gain.

Ported from the reference implementation -- the defaults the settled
method uses, with the expressions kept operation for operation so results match bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from appfl.decentralized.algorithm.adko_v2.memory import TokenMemory

#: The settled bandwidth for each kernel.
DEFAULT_BANDWIDTH = {"euclidean": 1.0, "hamming": 0.5}
KERNELS = tuple(DEFAULT_BANDWIDTH)


@dataclass
class ReasoningWeights:
    """Hyperparameters of the v2 score. The defaults are the settled method's.

    Args:
        beta: weight on my uncertainty ``sigma``.
        lam: weight on peer attraction ``G``.
        gamma: weight on peer repulsion ``Lambda``.
        kernel: ``"euclidean"`` for continuous inputs, ``"hamming"`` for categorical ones.
        bandwidth: the kernel's ``sigma``; ``None`` takes the settled value for the kernel.
        eps: floor on ``RMS_Z(G0 + P0)``, so a near-empty field is not blown up to full size.
    """

    beta: float = 2.0
    lam: float = 2.0
    gamma: float = 2.0
    kernel: str = "euclidean"
    bandwidth: Optional[float] = None
    eps: float = 0.05

    def __post_init__(self) -> None:
        if self.kernel not in KERNELS:
            raise ValueError(f"kernel must be one of {KERNELS}, got {self.kernel!r}")
        if self.bandwidth is None:
            self.bandwidth = DEFAULT_BANDWIDTH[self.kernel]
        if self.bandwidth <= 0 or self.eps <= 0:
            raise ValueError("bandwidth and eps must be positive")


def similarity(A, B, kind: str = "euclidean", bandwidth: float = 1.0) -> np.ndarray:
    """``(n, m)`` matrix ``S`` between rows: ``exp(-d^2/bw^2)`` or ``exp(-h^2/(2 bw^2))``."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    bw2 = float(bandwidth) ** 2
    if kind == "euclidean":
        d2 = (
            np.sum(A * A, axis=1)[:, None]
            + np.sum(B * B, axis=1)[None, :]
            - 2.0 * A @ B.T
        )
        return np.exp(-np.maximum(d2, 0.0) / bw2)
    if kind == "hamming":
        h = (A[:, None, :] != B[None, :, :]).mean(axis=2)
        return np.exp(-(h**2) / (2.0 * bw2))
    raise ValueError("kernel must be 'euclidean' or 'hamming'")


def source_weights(src, weights: Optional[Mapping[int, float]] = None) -> np.ndarray:
    """Per-token sender weight, normalized over the senders present in memory.

    ``weights`` maps a source index to its weight (trust's odds); sources it omits, or gives a
    non-positive weight, are inactive. ``None`` weighs every sender equally.
    """
    src = np.asarray(src, dtype=int)
    if weights is None:
        w = np.ones(len(src))
    else:
        w = np.array([max(float(weights.get(int(j), 0.0)), 0.0) for j in src])
    total = sum({int(j): wj for j, wj in zip(src, w)}.values())
    return w / total if total > 0 else np.zeros(len(src))


def field_coefficients(
    src, q, weights: Optional[Mapping[int, float]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-token attraction and repulsion coefficients ``(c+, c-)``.

    ``c+_k = w_j [q_k]_+ / |K_j|`` and ``c-_k = w_j [-q_k]_+ / |K_j|`` for ``k`` from source
    ``j``: each source's tokens are averaged, then sources are mixed by ``w``.
    """
    src = np.asarray(src, dtype=int)
    q = np.asarray(q, dtype=float)
    if src.size == 0:
        return np.zeros(0), np.zeros(0)
    w = source_weights(src, weights)
    m = np.abs(q)
    pos, neg = np.where(q > 0, m, 0.0), np.where(q < 0, m, 0.0)
    _, inv, counts = np.unique(src, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    denom = counts[inv].astype(float)
    denom = np.where(denom > 0, denom, 1.0)
    return w * pos / denom, w * neg / denom


class PeerField:
    """One agent's ``(G0, P0)`` for the round, with memory and weights frozen."""

    def __init__(
        self, memory: TokenMemory, weights: Optional[Mapping[int, float]] = None
    ):
        self.kind, self.bw = memory.kernel, memory.bandwidth
        self.X = memory.X
        self.cp, self.cm = field_coefficients(memory.src, memory.q, weights)

    def __call__(self, X) -> Tuple[np.ndarray, np.ndarray]:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self.cp.size == 0:
            return np.zeros(len(X)), np.zeros(len(X))
        S = similarity(X, self.X, self.kind, self.bw)
        return S @ self.cp, S @ self.cm


def _rms(v) -> float:
    v = np.asarray(v, dtype=float)
    return float(np.sqrt(np.mean(v * v))) if v.size else 0.0


class Calibrator:
    """``match_rms``: one factor ``k`` per round, fixed on the reference set ``Z``.

    ``k = RMS_Z(sigma) / max(RMS_Z(G0 + P0), eps)``, and ``(G, Lambda) = (k G0, k P0)``.
    """

    mode = "match_rms"

    def __init__(self, sigma_Z, G0_Z, P0_Z, eps: float = 0.05):
        sigma_Z, G0_Z, P0_Z = (
            np.asarray(v, dtype=float) for v in (sigma_Z, G0_Z, P0_Z)
        )
        self.eps = float(eps)
        U = _rms(sigma_Z)
        social = _rms(G0_Z + P0_Z)
        self.k = U / max(social, self.eps) if U > 0 else 0.0
        #: Logged once per round; ``floor_active`` means the field was too weak to match.
        self.diagnostics: Dict[str, object] = {
            "mode": self.mode,
            "U": U,
            "floor_active": social < self.eps,
            "k": self.k,
        }

    def apply(self, G0, P0) -> Tuple[np.ndarray, np.ndarray]:
        """``(G, Lambda)`` for candidates, with this round's frozen factor."""
        G0, P0 = np.asarray(G0, dtype=float), np.asarray(P0, dtype=float)
        return self.k * G0, self.k * P0


def score(mu, sigma, G, Lam, beta: float = 2.0, lam: float = 2.0, gamma: float = 2.0):
    """``R = mu + beta*sigma + lam*G - gamma*Lambda`` (larger is better)."""
    return mu + beta * sigma + lam * G - gamma * Lam
