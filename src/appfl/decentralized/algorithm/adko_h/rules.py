"""The numeric rules a campaign may ask for, and the two terms that apply to all of them.

Inside a campaign the planner is out of the loop: a rule scores the candidate pool and the
argmax inside the region is run. Four rules are ADKO v2's acquisition read as a choice rather
than a constant -- the settled social score, GP-UCB with the social term switched off, pure
exploitation, pure exploration -- and the fifth is the one a diversity problem needs:

* ``novelty`` -- the candidate furthest from everything anyone has run. Distances are taken in
  the shared embedding, over this agent's own points *and* the inputs on its peers' tokens, so
  it is a decentralized max-min (DPP-like) rule: no central set, just whatever reached me.

Two terms apply whatever the rule:

* the **claim penalty**, which pushes a candidate out of a region a neighbour has announced;
* an optional **novelty bonus**, which is what makes the diversity-aware ADKO baseline -- the
  arm that gets the diversity pressure without any language model, and therefore the arm the
  LLM has to beat for the method to be worth its tokens.

Both are scaled by ``RMS(sigma)`` over the pool, as the v2 calibration scales the social field:
a weight then means the same thing whatever the outcome's units, which is what lets one
configuration run over Suzuki yields, molecule scores and hull volumes.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from appfl.decentralized.algorithm.adko_h.campaign import RULES


def _rms(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=float)
    return float(np.sqrt(np.mean(v * v))) if v.size else 0.0


def nearest_distance(X: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Distance from each row of ``X`` to the nearest row of ``known``; ``inf`` if none.

    The novelty rule's raw signal. Chunked so a large pool against a long history stays within
    a few hundred megabytes, since this runs every round on every agent.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    K = np.atleast_2d(np.asarray(known, dtype=float))
    if K.size == 0 or K.shape[1] != X.shape[1]:
        return np.full(len(X), np.inf)
    out = np.empty(len(X))
    step = max(int(2_000_000 / max(len(K), 1)), 1)
    for a in range(0, len(X), step):
        b = min(a + step, len(X))
        d2 = (
            np.sum(X[a:b] ** 2, axis=1)[:, None]
            + np.sum(K**2, axis=1)[None, :]
            - 2.0 * X[a:b] @ K.T
        )
        out[a:b] = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
    return out


def values(
    rule: str,
    mu: np.ndarray,
    sigma: np.ndarray,
    G: np.ndarray,
    Lam: np.ndarray,
    novelty: Optional[np.ndarray] = None,
    *,
    beta: float = 2.0,
    lam: float = 2.0,
    gamma: float = 2.0,
) -> np.ndarray:
    """The rule's value on each candidate; larger is better everywhere.

    ``novelty`` is required only by the ``novelty`` rule and is normalized by its own spread,
    so it lands in the same range as the standardized ``mu`` the other rules return.
    """
    mu, sigma = np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float)
    G, Lam = np.asarray(G, dtype=float), np.asarray(Lam, dtype=float)
    if rule == "indep":
        return mu + beta * sigma
    if rule == "exploit":
        return mu.copy()
    if rule == "explore":
        return sigma.copy()
    if rule == "novelty":
        return _scaled_novelty(novelty, len(mu))
    return mu + beta * sigma + lam * G - gamma * Lam


def _scaled_novelty(novelty: Optional[np.ndarray], n: int) -> np.ndarray:
    """Novelty distances on a standardized scale, with ``inf`` (nothing known) as the maximum."""
    if novelty is None:
        return np.zeros(n)
    d = np.asarray(novelty, dtype=float).copy()
    finite = np.isfinite(d)
    if not finite.any():
        return np.zeros(n)
    d[~finite] = d[finite].max()
    s = _rms(d)
    return d / s if s > 0 else np.zeros(n)


def adjust(
    v: np.ndarray,
    sigma: np.ndarray,
    *,
    penalty: Optional[np.ndarray] = None,
    penalty_weight: float = 0.0,
    novelty: Optional[np.ndarray] = None,
    novelty_weight: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Apply the claim penalty and the novelty bonus to a rule's values.

    Returns the adjusted values and what each term contributed, which the run record keeps so
    an ablation can be read off a finished run instead of re-run.
    """
    v = np.asarray(v, dtype=float).copy()
    u = _rms(sigma)
    parts = {"scale": u, "penalty": 0.0, "novelty": 0.0}
    if penalty is not None and penalty_weight:
        term = float(penalty_weight) * u * np.asarray(penalty, dtype=float)
        v -= term
        parts["penalty"] = float(np.mean(term))
    if novelty is not None and novelty_weight:
        term = float(novelty_weight) * u * _scaled_novelty(novelty, len(v))
        v += term
        parts["novelty"] = float(np.mean(term))
    return v, parts


def pick(v: np.ndarray, mask: Optional[np.ndarray] = None) -> int:
    """Index of the best candidate, inside ``mask`` when one is given.

    A region that excludes everything in the pool is not an error: the planner is allowed to
    be wrong about where candidates exist, and the campaign still has to spend its budget, so
    the rule falls back to the whole pool and the run record notes it.
    """
    v = np.asarray(v, dtype=float)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.any():
            idx = np.flatnonzero(m)
            return int(idx[int(np.argmax(v[idx]))])
    return int(np.argmax(v))


def valid_rule(name: str) -> str:
    """``name`` if it is a rule, else ``"adko"`` -- the planner's output lands here too."""
    return str(name) if str(name) in RULES else "adko"
