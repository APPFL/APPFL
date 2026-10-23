"""Regions: the part of the design space a campaign commits to.

ADKO-H's planner does not pick experiments, it picks *where to look*: a region, a rule, a
budget, and a test it can be wrong about. A region therefore has to be small enough to state
in one line of a prompt, checkable on a candidate pool with one array operation, and
problem-agnostic -- the same planner runs over Suzuki's categorical conditions, a molecule
pool's property windows and a simulator's continuous input box.

So a problem declares a **region space**: a few named, human-meaningful coordinates
(``logP``, ``temperature``, ``solvent``), each continuous with bounds or categorical with options, plus
a map from a candidate point to its coordinates. A region is then a box over the continuous
coordinates and a subset over the categorical ones, and masking a pool is a column compare.

The region space is deliberately *not* the search space: molecules come from a fixed pool of
250k fingerprints, but "3 or more aromatic rings and logP in [2, 4]" is a region a language
model can reason about and a hypothesis can be judged against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

#: A continuous coordinate, held as a closed interval.
KIND_REAL = "real"
#: A categorical coordinate, held as a subset of option indices.
KIND_CAT = "cat"


@dataclass(frozen=True)
class Dim:
    """One coordinate of the region space.

    Args:
        name: short identifier, used in prompts and in a region's label.
        kind: :data:`KIND_REAL` or :data:`KIND_CAT`.
        lo: lower bound of a real coordinate (inclusive).
        hi: upper bound of a real coordinate (inclusive).
        options: option names of a categorical coordinate; its codes are their indices.
        meaning: one clause of plain text for the prompt, e.g. "lipophilicity, higher is
            greasier". The planner sees this and nothing else about the coordinate.
    """

    name: str
    kind: str = KIND_REAL
    lo: float = 0.0
    hi: float = 1.0
    options: Tuple[str, ...] = ()
    meaning: str = ""

    def __post_init__(self) -> None:
        if self.kind not in (KIND_REAL, KIND_CAT):
            raise ValueError(f"kind must be {KIND_REAL!r} or {KIND_CAT!r}, got {self.kind!r}")
        if self.kind == KIND_CAT and len(self.options) < 2:
            raise ValueError(f"categorical dim {self.name!r} needs at least two options")
        if self.kind == KIND_REAL and not self.hi > self.lo:
            raise ValueError(f"real dim {self.name!r} needs hi > lo, got [{self.lo}, {self.hi}]")

    @property
    def span(self) -> float:
        return len(self.options) if self.kind == KIND_CAT else float(self.hi - self.lo)

    def describe(self) -> str:
        if self.kind == KIND_CAT:
            body = "one of " + ", ".join(f"{i}={o}" for i, o in enumerate(self.options))
        else:
            body = f"{self.lo:g} to {self.hi:g}"
        return f"{self.name}: {body}" + (f" -- {self.meaning}" if self.meaning else "")


@dataclass(frozen=True)
class Region:
    """A box over the real coordinates and a subset over the categorical ones.

    A coordinate the region omits is unconstrained, so ``Region()`` is the whole space. Built
    through :meth:`RegionSpace.region`, which validates and clips against the declared bounds;
    the planner's output goes through the same door, so a malformed region becomes a wide one
    rather than an exception.
    """

    bounds: Mapping[str, Tuple[float, float]] = field(default_factory=dict)
    picks: Mapping[str, Tuple[int, ...]] = field(default_factory=dict)

    def mask(self, space: "RegionSpace", F: np.ndarray) -> np.ndarray:
        """Boolean mask over the rows of ``F``, the pool's region coordinates."""
        F = np.atleast_2d(np.asarray(F, dtype=float))
        ok = np.ones(len(F), dtype=bool)
        for name, (lo, hi) in self.bounds.items():
            col = F[:, space.index[name]]
            ok &= (col >= lo) & (col <= hi)
        for name, opts in self.picks.items():
            col = np.rint(F[:, space.index[name]]).astype(np.int64)
            ok &= np.isin(col, np.asarray(opts, dtype=np.int64))
        return ok

    def share(self, space: "RegionSpace") -> float:
        """Fraction of the space's volume, as a product over coordinates.

        A crude prior, but the planner is asked for regions it can justify, and a region that
        covers 90% of the space is not a hypothesis. Logged, and used by the random control to
        draw designs of comparable size.
        """
        out = 1.0
        for d in space.dims:
            if d.kind == KIND_CAT:
                n = len(self.picks.get(d.name, range(len(d.options))))
                out *= n / len(d.options)
            else:
                lo, hi = self.bounds.get(d.name, (d.lo, d.hi))
                out *= max(min(hi, d.hi) - max(lo, d.lo), 0.0) / d.span
        return float(out)

    def overlap(self, space: "RegionSpace", other: "Region") -> float:
        """Shared fraction of the two regions, per coordinate and multiplied.

        How a claim steers a neighbour away: overlap near 1 means we would be running the same
        experiment, overlap 0 means the claim says nothing about where I am looking.
        """
        out = 1.0
        for d in space.dims:
            if d.kind == KIND_CAT:
                a = set(self.picks.get(d.name, range(len(d.options))))
                b = set(other.picks.get(d.name, range(len(d.options))))
                inter, union = len(a & b), len(a | b)
                out *= inter / union if union else 1.0
            else:
                alo, ahi = self.bounds.get(d.name, (d.lo, d.hi))
                blo, bhi = other.bounds.get(d.name, (d.lo, d.hi))
                inter = max(min(ahi, bhi) - max(alo, blo), 0.0)
                union = max(ahi, bhi) - min(alo, blo)
                out *= inter / union if union > 0 else 1.0
        return float(out)

    def label(self, space: "RegionSpace") -> str:
        """One line, as the prompts and the run record show it."""
        parts: List[str] = []
        for d in space.dims:
            if d.kind == KIND_CAT and d.name in self.picks:
                names = [d.options[i] for i in self.picks[d.name] if i < len(d.options)]
                parts.append(f"{d.name}{{{','.join(names)}}}")
            elif d.kind == KIND_REAL and d.name in self.bounds:
                lo, hi = self.bounds[d.name]
                parts.append(f"{d.name}[{lo:g},{hi:g}]")
        return " ".join(parts) if parts else "whole space"

    def to_dict(self) -> Dict[str, Any]:
        """The readable form, for run records and prompts."""
        return {
            "bounds": {k: [float(v[0]), float(v[1])] for k, v in self.bounds.items()},
            "picks": {k: [int(i) for i in v] for k, v in self.picks.items()},
        }

    def pack(self, space: "RegionSpace") -> List[Any]:
        """The wire form: one entry per coordinate in the space's order, ``None`` for free.

        A claim crosses an agent boundary every round it is relayed, and the named form costs
        several times the token it rides on -- measured, not assumed. Positional entries put
        the campaign channel back in the same order of magnitude as the result it accompanies.
        """
        out: List[Any] = []
        for d in space.dims:
            if d.kind == KIND_CAT:
                v = self.picks.get(d.name)
                out.append([int(i) for i in v] if v else None)
            else:
                v = self.bounds.get(d.name)
                out.append([round(float(v[0]), 4), round(float(v[1]), 4)] if v else None)
        return out


class RegionSpace:
    """The coordinates a problem exposes to the planner, and the pool's values on them.

    Args:
        dims: the coordinates, in a fixed order -- the column order of every ``F`` array.
    """

    def __init__(self, dims: Sequence[Dim]):
        if not dims:
            raise ValueError("a region space needs at least one dimension")
        self.dims: Tuple[Dim, ...] = tuple(dims)
        self.index: Dict[str, int] = {d.name: i for i, d in enumerate(self.dims)}
        if len(self.index) != len(self.dims):
            raise ValueError("dimension names must be unique")

    def __len__(self) -> int:
        return len(self.dims)

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(d.name for d in self.dims)

    def describe(self) -> str:
        """The block of text the prompt uses to explain the space."""
        return "\n".join(f"- {d.describe()}" for d in self.dims)

    def region(
        self,
        bounds: Optional[Mapping[str, Sequence[float]]] = None,
        picks: Optional[Mapping[str, Sequence[int]]] = None,
    ) -> Region:
        """A validated region: unknown names dropped, intervals clipped and ordered.

        Forgiving on purpose -- this is where a language model's JSON lands.
        """
        b: Dict[str, Tuple[float, float]] = {}
        for name, pair in (bounds or {}).items():
            d = self._dim(name, KIND_REAL)
            if d is None or pair is None or len(tuple(pair)) != 2:
                continue
            lo, hi = sorted(float(v) for v in pair)
            lo, hi = max(lo, d.lo), min(hi, d.hi)
            if hi > lo:
                b[name] = (lo, hi)
        p: Dict[str, Tuple[int, ...]] = {}
        for name, opts in (picks or {}).items():
            d = self._dim(name, KIND_CAT)
            if d is None or opts is None:
                continue
            keep = sorted({int(i) for i in opts if 0 <= int(i) < len(d.options)})
            if keep:
                p[name] = tuple(keep)
        return Region(bounds=b, picks=p)

    def _dim(self, name: str, kind: str) -> Optional[Dim]:
        i = self.index.get(str(name))
        if i is None:
            return None
        d = self.dims[i]
        return d if d.kind == kind else None

    def full(self) -> Region:
        return Region()

    def unpack(self, data: Optional[Sequence[Any]]) -> Region:
        """Rebuild a region from :meth:`Region.pack`, validated like any other input."""
        bounds: Dict[str, Sequence[float]] = {}
        picks: Dict[str, Sequence[int]] = {}
        for d, v in zip(self.dims, list(data or [])):
            if v is None:
                continue
            if d.kind == KIND_CAT:
                picks[d.name] = [int(i) for i in v]
            elif len(tuple(v)) >= 2:
                bounds[d.name] = (float(v[0]), float(v[1]))
        return self.region(bounds=bounds, picks=picks)

    # -- territories ---------------------------------------------------------------------

    def normalize(self, F: np.ndarray) -> np.ndarray:
        """``F`` mapped to the unit cube, so distances treat every coordinate alike."""
        F = np.atleast_2d(np.asarray(F, dtype=float))
        out = np.empty_like(F, dtype=float)
        for i, d in enumerate(self.dims):
            if d.kind == KIND_CAT:
                out[:, i] = np.clip(F[:, i], 0, len(d.options) - 1) / max(len(d.options) - 1, 1)
            else:
                out[:, i] = np.clip((F[:, i] - d.lo) / d.span, 0.0, 1.0)
        return out

    def centers(self, n: int, seed: int) -> np.ndarray:
        """``n`` spread points in the unit cube: the agents' territory seeds.

        Sobol when scipy is present, uniform otherwise. Only the first campaign of a run is
        confined to a territory, so an imperfect spread costs a little balance, not validity.
        """
        try:
            from scipy.stats import qmc

            return np.asarray(qmc.Sobol(len(self.dims), scramble=True, seed=seed).random(n))
        except Exception:
            return np.random.default_rng(seed).random((n, len(self.dims)))

    def cell_mask(self, F: np.ndarray, centers: np.ndarray, which: int) -> np.ndarray:
        """Rows of ``F`` whose nearest center is ``which`` -- one agent's Voronoi cell."""
        U = self.normalize(F)
        C = np.atleast_2d(np.asarray(centers, dtype=float))
        d = ((U[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        return np.argmin(d, axis=1) == int(which)

    def box_around(
        self, center: Sequence[float], half_width: Union[float, Sequence[float]] = 0.25
    ) -> Region:
        """A region of the given half-width (in normalized units) about a unit-cube point.

        How a territory becomes something the planner can read and narrow: the first campaign
        is handed this box rather than a bare Voronoi cell. ``half_width`` is one number for
        every coordinate, or one per coordinate in the space's order -- which is how a control
        draws boxes shaped like the ones a planner actually chose.
        """
        c = np.asarray(center, dtype=float)
        hw = np.broadcast_to(np.asarray(half_width, dtype=float), (len(self.dims),))
        bounds: Dict[str, Sequence[float]] = {}
        picks: Dict[str, Sequence[int]] = {}
        for i, d in enumerate(self.dims):
            u = float(np.clip(c[i] if i < len(c) else 0.5, 0.0, 1.0))
            h = float(hw[i])
            if d.kind == KIND_CAT:
                k = int(round(u * (len(d.options) - 1)))
                span = max(int(round(h * len(d.options))), 1)
                picks[d.name] = list(range(max(k - span, 0), min(k + span + 1, len(d.options))))
            else:
                mid = d.lo + u * d.span
                bounds[d.name] = (mid - h * d.span, mid + h * d.span)
        return self.region(bounds=bounds, picks=picks)
