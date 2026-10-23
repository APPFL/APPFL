"""Endpoints, computed from a finished run's record rather than inside the loop.

Keeping them out of the loop is deliberate: an arm that steers on the same quantity it is
scored by has an advantage the numeric baselines do not, and the honest comparison reports
both -- what each arm optimized and what every arm is measured on. So these functions take a
run record's columns and nothing else, and the same functions score every arm.

Four endpoints:

* ``cumulative_distinct`` -- distinct family keys against evaluations. The count endpoint.
* ``n_circles`` -- the size of a maximal set of outcomes pairwise further apart than a
  threshold (Xie et al., ICLR 2023), which is the count endpoint without a classifier.
* ``hull_volume`` -- the convex hull of outcomes in a frozen descriptor basis.
* ``auc_top_k`` -- the optimization endpoint, as PMO defines it, so a diversity claim cannot
  quietly be a claim about having given up on the objective.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


def cumulative_distinct(families: Sequence[str]) -> List[int]:
    """Running count of distinct non-empty keys; every entry spends one unit of budget."""
    seen: set = set()
    out: List[int] = []
    for f in families:
        if f:
            seen.add(f)
        out.append(len(seen))
    return out


def n_circles(Z: np.ndarray, threshold: float) -> int:
    """Greedy count of outcomes pairwise at least ``threshold`` apart.

    The standard greedy pass: take each point in order, keep it if it clears every point kept
    so far. Order-dependent and a lower bound on the true packing number, which is what the
    literature reports.
    """
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    if Z.size == 0:
        return 0
    kept = [0]
    for i in range(1, len(Z)):
        d = np.sqrt(((Z[kept] - Z[i]) ** 2).sum(axis=1))
        if d.min() >= threshold:
            kept.append(i)
    return len(kept)


def hull_volume(Z: np.ndarray) -> float:
    """Convex-hull volume of the outcomes, or 0.0 when they are degenerate or too few."""
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    if Z.ndim != 2 or len(Z) <= Z.shape[1]:
        return 0.0
    try:
        from scipy.spatial import ConvexHull, QhullError

        try:
            return float(ConvexHull(Z).volume)
        except QhullError:
            return 0.0
    except ImportError:
        return 0.0


def auc_top_k(scores: Sequence[float], k: int = 10, budget: Optional[int] = None) -> float:
    """PMO's AUC top-``k``: the mean over the budget of the running top-``k`` average.

    Rewards reaching good molecules early, not merely reaching them. A run shorter than
    ``budget`` is padded with its final value, as PMO does, so arms that stop early are not
    flattered.
    """
    s = np.asarray(list(scores), dtype=float)
    if s.size == 0:
        return 0.0
    running: List[float] = []
    best: List[float] = []
    for v in s:
        best.append(float(v))
        best.sort(reverse=True)
        del best[k:]
        running.append(float(np.mean(best)))
    if budget and len(running) < budget:
        running += [running[-1]] * (budget - len(running))
    return float(np.mean(running))


def diverse_hits(
    scores: Sequence[float], families: Sequence[str], threshold: float
) -> int:
    """Distinct families among outcomes scoring at least ``threshold``.

    The "diverse hits" endpoint (Renz et al., JCIM 2024): a method that finds one good family a
    thousand times scores 1, which is the failure the plain optimization endpoints hide.
    """
    s = np.asarray(list(scores), dtype=float)
    return len({f for f, v in zip(families, s) if f and v >= threshold})


def brier(p: Sequence[float], outcome: Sequence[bool]) -> Optional[float]:
    """Mean ``(p - y)^2`` of the planner's stated probabilities."""
    p = np.asarray(list(p), dtype=float)
    y = np.asarray([1.0 if o else 0.0 for o in outcome], dtype=float)
    return float(np.mean((p - y) ** 2)) if p.size else None


def calibration(p: Sequence[float], outcome: Sequence[bool]) -> Dict[str, Optional[float]]:
    """The planner's Brier score against the only baseline that matters: its own base rate.

    A planner that says 0.7 to everything and is right 70% of the time has learned nothing
    about *which* campaigns work, and ``skill <= 0`` says so.
    """
    y = [1.0 if o else 0.0 for o in outcome]
    if not y:
        return {"brier": None, "base_rate": None, "reference": None, "skill": None}
    rate = float(np.mean(y))
    b = brier(p, outcome)
    ref = brier([rate] * len(y), outcome)
    skill = None if not ref else 1.0 - (b / ref)
    return {"brier": b, "base_rate": rate, "reference": ref, "skill": skill}


def load_balance(agents: Sequence[int]) -> Dict[str, float]:
    """How evenly the budget was spent across agents -- the cost of territories, if any."""
    if not len(agents):
        return {"gini": 0.0, "min": 0.0, "max": 0.0}
    counts = np.bincount(np.asarray(agents, dtype=int))
    counts = counts[counts > 0].astype(float) if counts.size else counts.astype(float)
    n = len(counts)
    if n == 0:
        return {"gini": 0.0, "min": 0.0, "max": 0.0}
    srt = np.sort(counts)
    gini = float((2.0 * np.sum((np.arange(1, n + 1)) * srt) / (n * np.sum(srt))) - (n + 1) / n)
    return {"gini": gini, "min": float(srt[0]), "max": float(srt[-1])}


def redundancy(X: np.ndarray, threshold: float) -> int:
    """Pairs of evaluated points closer than ``threshold`` -- wasted, uncoordinated effort."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if len(X) < 2:
        return 0
    total = 0
    for a in range(0, len(X), 512):
        b = min(a + 512, len(X))
        d2 = (
            np.sum(X[a:b] ** 2, axis=1)[:, None]
            + np.sum(X**2, axis=1)[None, :]
            - 2.0 * X[a:b] @ X.T
        )
        d = np.sqrt(np.maximum(d2, 0.0))
        for i in range(b - a):
            total += int(np.count_nonzero(d[i, a + i + 1 :] < threshold))
    return total
