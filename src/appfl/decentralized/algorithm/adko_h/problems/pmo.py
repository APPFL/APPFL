"""PMO: 250k purchasable molecules, one oracle, and diversity as the endpoint.

Why this proxy. Suzuki's 3,696 conditions are small enough that every agent scores the whole
space every round, which is the one thing a large problem never allows -- and the L3 screen
showed how much that matters: with the whole table enumerable, a planner has almost nothing to
add. PMO (Gao et al., NeurIPS 2022 Datasets & Benchmarks) is the opposite and is standard:
a quarter of a million candidates, 20 goal-directed oracles with published numbers, a fixed
call budget, and an established diversity literature on top of it (``#Circles``, Xie et al.
ICLR 2023; diverse hits, Renz et al. JCIM 2024). It is cheap, it is CPU-only, and a reviewer
has seen it before.

How it maps onto ADKO-H:

* **the pool** is the design space; every round each agent draws its own random subpool, so no
  agent ever sees the whole space -- the regime large problems are in;
* **the region space** is five chemist-meaningful properties (molecular weight, logP, TPSA,
  rings, QED), so a campaign is a sentence a chemist would recognize: "polar, ring-rich, 300
  to 400 Da";
* **the family key** is the Bemis-Murcko scaffold, which is what "a different kind of
  molecule" means in this field, and is exactly the kind of qualitative class key the family
  tests are written for;
* **the descriptor** is a frozen three-dimensional projection of the fingerprint, so hull
  growth and novelty distance are defined as for any continuous descriptor.

One simplification, stated rather than buried: the shared embedding is a frozen random
projection of the ECFP4 fingerprint, not the fingerprint itself. A token has to stay small,
and the GP, the peer kernel and the token all read the same vector in ADKO. A Tanimoto GP on
the full fingerprint would model better; every arm here pays the same cost, so the comparison
between arms is unaffected, and absolute AUC sits below the published leaderboard.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from appfl.decentralized.algorithm.adko_h.problem import Candidates, Outcome, Pick, Problem
from appfl.decentralized.algorithm.adko_h.problems.pmo_oracles import get_oracle, murcko_scaffold
from appfl.decentralized.algorithm.adko_h.problems.pmo_oracles.pool import Pool, descriptors
from appfl.decentralized.algorithm.adko_h.region import Dim, RegionSpace

#: Where the pool file lives, when no path is passed.
POOL_ENV = "ADKO_PMO_POOL"
#: Region coordinates, with the bounds the ZINC pool actually spans.
PROPERTIES = (
    ("mw", 150.0, 550.0, "molecular weight in daltons"),
    ("logp", -2.0, 7.0, "lipophilicity; higher is greasier, lower is more polar"),
    ("tpsa", 0.0, 180.0, "topological polar surface area; polarity and H-bonding"),
    ("num_rings", 0.0, 7.0, "ring count; higher is more rigid and more scaffold-like"),
    ("qed", 0.0, 1.0, "drug-likeness, 0 to 1"),
)


class PMO(Problem):
    """One PMO oracle over a fixed molecule pool.

    Args:
        oracle: a PMO task name, e.g. ``"albuterol_similarity"`` or ``"zaleplon_mpo"``.
        pool_path: the ``.smi`` file; defaults to ``$ADKO_PMO_POOL``.
        limit: use only the first ``limit`` molecules (for tests and smoke runs).
        embed_dim: width of the frozen fingerprint projection the agents share.
        seed: fixes the projection and the descriptor basis, so every agent and every arm of a
            comparison embeds a molecule identically.
        blind: describe the objective to the planner without naming it. A model that knows
            the benchmark can aim at its answer; a real application offers no such prior, and
            this is how to measure what that knowledge was worth.
    """

    maximize = True
    kernel = "euclidean"

    def __init__(
        self,
        oracle: str = "albuterol_similarity",
        pool_path: Optional[str] = None,
        limit: Optional[int] = None,
        embed_dim: int = 32,
        seed: int = 0,
        verbose: bool = False,
        blind: bool = False,
    ):
        path = Path(pool_path or os.environ.get(POOL_ENV, ""))
        if not path.is_file():
            raise FileNotFoundError(
                f"molecule pool not found: set {POOL_ENV} or pass pool_path (got {path!r}). "
                f"Build one with python -m ...problems.pmo_oracles.pool / fetch_pool.py."
            )
        self.problem_id = f"pmo:{oracle}"
        self.oracle_name = str(oracle)
        self._oracle = get_oracle(self.oracle_name)
        self.pool_data = Pool.load(path, limit=limit, verbose=verbose)
        self.smiles: List[str] = list(self.pool_data.smiles)
        self.n = len(self.smiles)
        self.region_space = RegionSpace(
            [Dim(k, "real", lo, hi, meaning=m) for k, lo, hi, m in PROPERTIES]
        )
        self._F, self._X, self._Z = self._tables(int(embed_dim), int(seed))
        self._x_scale = np.maximum(self._X.std(axis=0), 1e-12)
        self.blind = bool(blind)
        self._scaffolds: Dict[int, str] = {}
        self._cache: Dict[int, float] = {}
        #: Oracle calls, which is what PMO charges a method for.
        self.calls = 0

    # -- fixed, frozen tables ---------------------------------------------------------------

    def _features(self) -> np.ndarray:
        """The pool's region coordinates. NaNs (unparseable) are pushed out of every region."""
        d = descriptors(self.smiles)
        F = np.column_stack([np.asarray(d[k], dtype=float) for k, *_ in PROPERTIES])
        bad = ~np.isfinite(F).all(axis=1)
        F[bad] = np.inf
        return F

    def _tables(self, embed_dim: int, seed: int):
        """Region coordinates, the shared embedding and the descriptor basis -- all cached.

        These three tables are identical for every arm, seed and agent, and they are not
        cheap: rdkit descriptors over 250k molecules take about five minutes and the
        projection another few. A job array would otherwise pay that per process, so they are
        computed once and cached next to the pool. Build the cache on a login node with
        ``setup.sh``; a compute node then starts in seconds.
        """
        cache = Path(self.pool_data.source).with_name(
            f"{Path(self.pool_data.source).stem}.tab-n{self.n}-d{embed_dim}-s{seed}.npz"
        )
        if cache.is_file():
            try:
                with np.load(cache, allow_pickle=False) as d:
                    if int(d["n"]) == self.n and int(d["dim"]) == embed_dim:
                        return d["F"], d["X"], d["Z"]
            except (OSError, KeyError, ValueError):
                pass
        F = self._features()
        X, Z = self._project(embed_dim, seed)
        try:
            np.savez_compressed(cache, F=F, X=X, Z=Z, n=self.n, dim=embed_dim, seed=seed)
        except OSError:
            pass  # a read-only pool directory is not a reason to fail a run
        return F, X, Z

    def _project(self, embed_dim: int, seed: int):
        rng = np.random.default_rng([seed, 4242])
        nbits = int(self.pool_data.nbits)
        P = rng.normal(size=(nbits, embed_dim)) / np.sqrt(nbits)
        out = np.zeros((self.n, embed_dim))
        for a in range(0, self.n, 8192):
            b = min(a + 8192, self.n)
            bits = np.unpackbits(self.pool_data.fps[a:b], axis=1)[:, :nbits].astype(np.float32)
            out[a:b] = bits @ P
        out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
        out *= np.sqrt(embed_dim) / 4.0  # distances of order one, so bandwidth 1 is sane
        Z = out[:, :3] - out[:, :3].mean(axis=0)
        s = np.maximum(Z.std(axis=0), 1e-9)
        return out, Z / s

    # -- the Problem surface ----------------------------------------------------------------

    def pool(
        self,
        rng: np.random.Generator,
        n: int,
        exclude: Optional[Sequence[int]] = None,
        region=None,
    ) -> Candidates:
        """A random subpool, drawn inside ``region`` when enough of the pool lies there.

        A fixed pool cannot sample a region the way a continuous space can, so this filters
        first and falls back to the whole pool when the region is too thin to fill a draw --
        the rule's mask then reports an empty region instead of a silently unrepresentative
        pool.
        """
        idx = np.arange(self.n)
        if region is not None:
            inside = region.mask(self.region_space, self._F)
            if int(np.count_nonzero(inside)) >= max(n, 16):
                idx = np.flatnonzero(inside)
        if exclude:
            idx = np.setdiff1d(idx, np.asarray(list(exclude), dtype=np.int64), assume_unique=False)
        if len(idx) == 0:
            idx = np.arange(self.n)
        take = rng.choice(idx, size=min(int(n), len(idx)), replace=False)
        take = np.sort(np.asarray(take, dtype=np.int64))
        return Candidates(
            key=take,
            X=self._X[take],
            F=self._F[take],
            meta=[self.smiles[int(i)] for i in take],
        )

    def start(self, seed: int, n: int) -> Candidates:
        """The shared initial molecules: a fixed random draw, the same for every arm."""
        if n <= 0:
            return super().start(seed, n)
        return self.pool(np.random.default_rng([int(seed), 1234]), int(n))

    def evaluate(self, picks: Sequence[Pick]) -> List[Outcome]:
        out: List[Outcome] = []
        for p in picks:
            key = int(p.key)
            smi = p.meta if isinstance(p.meta, str) else self.smiles[key]
            if key not in self._cache:
                self._cache[key] = float(self._oracle(smi))
                self.calls += 1
            if key not in self._scaffolds:
                self._scaffolds[key] = murcko_scaffold(smi)
            out.append(
                Outcome(
                    score=self._cache[key],
                    descriptor=self._Z[key],
                    family=self._scaffolds[key],
                )
            )
        return out

    def descriptor_dim(self) -> int:
        return 3

    def label(self, key: int, meta: Any = None) -> str:
        return meta if isinstance(meta, str) else self.smiles[int(key)]

    def input_scale(self) -> np.ndarray:
        """The frozen embedding's per-coordinate standard deviation over the whole pool."""
        return self._x_scale

    def describe(self) -> Dict[str, Any]:
        objective = (
            "a hidden objective"
            if self.blind
            else f"the PMO objective '{self.oracle_name}'"
        )
        hints = (
            "Properties are correlated: high logP tends to come with low TPSA, and QED "
            "falls off above about 500 daltons."
        )
        if not self.blind:
            hints += (
                " Similarity and rediscovery objectives "
                "reward a narrow region near the target's properties; MPO objectives reward "
                "hitting several property windows at once, so a region that fixes the easy "
                "properties and leaves one free is often more informative than a tight box."
            )
        return {
            "task": (
                f"Search a fixed pool of {self.n:,} purchasable molecules under "
                f"{objective}, which scores each molecule in [0, 1]. The "
                "run is judged on how many *distinct* scaffolds it finds at a high score, not "
                "on the single best molecule, so a region that yields one good series "
                "repeatedly is worth less than two regions yielding different series."
            ),
            "outcome": (
                "Each experiment returns the oracle score, a three-dimensional structural "
                "descriptor, and the molecule's Bemis-Murcko scaffold, which is the family "
                "key. A new family means a structurally different core."
            ),
            "hints": hints,
            "dims": self.region_space.describe(),
        }
