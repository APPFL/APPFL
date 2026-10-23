"""A free problem with the shape of an expensive one, for tests and for sizing a run.

Five continuous inputs, two of them fractions constrained to sum to at most one, a
three-dimensional outcome descriptor, and a family key that names which basin an outcome came
from rather than how good it was. That shape is what the method's machinery touches -- regions
over inputs, novelty in descriptor space, family counts, hull growth -- so a test here exercises
the same code paths an expensive simulator would, at zero cost.

The landscape has a handful of narrow basins, and the family key changes qualitatively between
them, which gives a diversity method something to find and a yield-chasing method something to
over-exploit. Nothing here is meant to be scientifically interesting; it is meant to make a
bug visible in two seconds instead of two GPU-hours.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from appfl.decentralized.algorithm.adko_h.problem import Candidates, Outcome, Pick, Problem
from appfl.decentralized.algorithm.adko_h.region import Dim, RegionSpace

#: Basin centers in the 5-D unit cube, their widths and their family labels.
#: Wide enough that a few dozen evaluations find several -- a test has to be able to fail for
#: the right reason, and a landscape nothing can hit makes every arm look identical.
BASINS = (
    ((0.15, 0.20, 0.30, 0.70, 0.50), 0.42, "alpha"),
    ((0.80, 0.15, 0.60, 0.25, 0.40), 0.38, "beta"),
    ((0.45, 0.40, 0.20, 0.50, 0.85), 0.34, "gamma"),
    ((0.60, 0.30, 0.80, 0.80, 0.20), 0.30, "delta"),
    ((0.25, 0.65, 0.50, 0.35, 0.65), 0.26, "epsilon"),
)


class Synthetic(Problem):
    """Five inputs, three descriptors, five findable families plus a dull background.

    Args:
        seed: fixes the noise stream and the pool draws.
        noise: observation noise on the score.
        dim: input dimension; the first two are treated as a simplex (they must sum to <= 1).
    """

    problem_id = "synthetic5"
    maximize = True
    kernel = "euclidean"

    def __init__(self, seed: int = 0, noise: float = 0.01, dim: int = 5):
        self.seed, self.noise, self.dim = int(seed), float(noise), int(dim)
        names = ["x1", "x2", "x3", "x4", "x5"][:dim]
        self.region_space = RegionSpace(
            [
                Dim(n, "real", 0.0, 1.0, meaning=m)
                for n, m in zip(
                    names,
                    [
                        "first fraction",
                        "second fraction (x1 + x2 <= 1)",
                        "first coupling",
                        "second coupling",
                        "third coupling",
                    ],
                )
            ]
        )
        self._next = 0
        self._points: Dict[int, np.ndarray] = {}

    # -- candidates ------------------------------------------------------------------------

    def pool(
        self,
        rng: np.random.Generator,
        n: int,
        exclude: Optional[Sequence[int]] = None,
        region=None,
    ) -> Candidates:
        """Fresh points each round: uniform in the cube, or inside ``region`` when given.

        Continuous problems can sample a region directly, which is what makes a narrow
        campaign cheap -- the alternative, masking a uniform pool, wastes the pool when the
        region is small.
        """
        X = self._draw(rng, int(n), region)
        keys = np.arange(self._next, self._next + len(X), dtype=np.int64)
        self._next += len(X)
        for k, x in zip(keys, X):
            self._points[int(k)] = x
        return Candidates(key=keys, X=X, F=X.copy(), meta=[tuple(np.round(x, 4)) for x in X])

    def _draw(self, rng: np.random.Generator, n: int, region) -> np.ndarray:
        lo = np.zeros(self.dim)
        hi = np.ones(self.dim)
        if region is not None:
            for name, (a, b) in getattr(region, "bounds", {}).items():
                i = self.region_space.index[name]
                lo[i], hi[i] = a, b
        keep = [self._feasible(rng.uniform(lo, hi, size=(int(n * 1.4) + 8, self.dim)))]
        for _ in range(3):  # a region can sit mostly outside the simplex
            if sum(len(k) for k in keep) >= n:
                break
            keep.append(self._feasible(rng.uniform(lo, hi, size=(n, self.dim))))
        X = np.vstack(keep)
        if len(X) < n:
            # The region misses the simplex. Fall back to the whole space and let the rule's
            # mask report an empty region rather than spinning here.
            X = np.vstack([X, self._feasible(rng.uniform(0.0, 1.0, size=(4 * n, self.dim)))])
        return np.ascontiguousarray(X[:n])

    @staticmethod
    def _feasible(X: np.ndarray) -> np.ndarray:
        return X[X[:, 0] + X[:, 1] <= 1.0]

    def start(self, seed: int, n: int) -> Candidates:
        if n <= 0:
            return super().start(seed, n)
        return self.pool(np.random.default_rng([seed, 99]), n)

    # -- evaluation ------------------------------------------------------------------------

    def evaluate(self, picks: Sequence[Pick]) -> List[Outcome]:
        out: List[Outcome] = []
        for p in picks:
            x = np.asarray(p.X, dtype=float).ravel()
            rng = np.random.default_rng([self.seed, int(p.key)])
            score, fam = self._landscape(x)
            score = float(np.clip(score + rng.normal(0.0, self.noise), 0.0, 1.0))
            out.append(Outcome(score=score, descriptor=self._descriptor(x, score), family=fam))
        return out

    def _landscape(self, x: np.ndarray):
        best, fam = 0.05 + 0.1 * float(np.mean(x)), ""
        for center, width, label in BASINS:
            c = np.asarray(center[: self.dim], dtype=float)
            d = float(np.linalg.norm(x - c))
            v = float(np.exp(-0.5 * (d / width) ** 2))
            if v > 0.45 and v > best:
                best, fam = 0.55 + 0.45 * v, label
        return best, fam or "background"

    def _descriptor(self, x: np.ndarray, score: float) -> np.ndarray:
        """Three outcome coordinates, as the real problem's frozen PC basis would give."""
        return np.asarray(
            [
                float(np.sin(3.0 * x[0]) + x[1]),
                float(np.cos(2.5 * x[2]) - 0.5 * x[0]),
                float(2.0 * score - 1.0),
            ]
        )

    def descriptor_dim(self) -> int:
        return 3

    def label(self, key: int, meta: Any = None) -> str:
        x = self._points.get(int(key))
        return "(" + ",".join(f"{v:.3f}" for v in x) + ")" if x is not None else str(key)

    def describe(self) -> Dict[str, Any]:
        return {
            "task": (
                "Find as many qualitatively different outcome families as possible in a "
                "five-input continuous design space. Each experiment returns a score, a "
                "three-dimensional outcome descriptor and a family label."
            ),
            "outcome": (
                "A family label names the qualitative class of the outcome. New families are "
                "what the run is scored on; the score alone is not the goal."
            ),
            "hints": (
                "x1 and x2 are fractions and must sum to at most 1. Families sit in narrow "
                "basins, so a region much wider than a tenth of the space will mostly return "
                "the background family."
            ),
            "dims": self.region_space.describe(),
        }
