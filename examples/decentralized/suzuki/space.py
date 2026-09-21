"""Suzuki-Miyaura search space and dataset-backed oracle.

Each design point is a tuple of integer category IDs:
``(electrophile, nucleophile, base, ligand, solvent)``. Those integers are the
actual representation used by the reference ADKO Suzuki code: the GP sees the
category IDs directly, and peer-token similarity uses Hamming distance on the
same IDs.

The oracle is the real Olympus ``suzuki_edbo`` lookup table. Evaluating a point
means looking up its measured reaction yield, not running a synthetic objective.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from appfl.decentralized.algorithm.adko import DesignSpace

# Ordered SMILES from the reference. The index is the category ID, so order matters.
PARAM_OPTIONS: Tuple[Tuple[str, ...], ...] = (
    (  # dim 0: electrophile -- 4 options
        "ClC1=CC=C(N=CC=C2)C2=C1",
        "BrC1=CC=C(N=CC=C2)C2=C1",
        "O=S(OC1=CC=C(N=CC=C2)C2=C1)(C(F)(F)F)=O",
        "IC1=CC=C(N=CC=C2)C2=C1",
    ),
    (  # dim 1: nucleophile -- 3 options
        "CC1=CC=C(N(C2CCCCO2)N=C3)C3=C1B(O)O",
        "CC1=CC=C(N(C2CCCCO2)N=C3)C3=C1B4OC(C)(C)C(C)(C)O4",
        "CC1=CC=C(N(C2CCCCO2)N=C3)C3=C1[B-](F)(F)F",
    ),
    (  # dim 2: base -- 7 options
        "[Na+].[OH-]",
        "OC([O-])=O.[Na+]",
        "[Cs+].[F-]",
        "O=P([O-])([O-])[O-].[K+].[K+].[K+]",
        "[K+].[OH-]",
        "CC([O-])C.[Li+]",
        "CCN(CC)CC",
    ),
    (  # dim 3: ligand -- 11 options
        "CC(P(C(C)(C)C)C(C)(C)C)(C)C",
        "P(C1=CC=CC=C1)(C2=CC=CC=C2)C3=CC=CC=C3",
        "CC(C)(C)P(C(C)(C)C)C1=CC=C(N(C)C)C=C1",
        "P(C1CCCCC1)(C2CCCCC2)C3CCCCC3",
        "CC1=CC=CC=C1P(C2=CC=CC=C2C)C3=CC=CC=C3C",
        "CCCCP(C12C[C@@H]3C[C@@H](C[C@H](C2)C3)C1)C45C[C@H]6C[C@@H](C5)C[C@@H](C4)C6",
        "COC1=CC=CC(OC)=C1C2=C(P(C3CCCCC3)C4CCCCC4)C=CC=C2",
        "CC(C)(P(C(C)(C)C)[c-]1cccc1)C.CC(C)(P(C(C)(C)C)[c-]2cccc2)C.[Fe+2]",
        "CC(C1=C(C2=CC=CC=C2P(C3CCCCC3)C4CCCCC4)C(C(C)C)=CC(C(C)C)=C1)C",
        "[c-]1(P(C2=CC=CC=C2)C3=CC=CC=C3)cccc1.[c-]4(P(C5=CC=CC=C5)C6=CC=CC=C6)cccc4.[Fe+2]",
        "CC1(C)C2=C(OC3=C1C=CC=C3P(C4=CC=CC=C4)C5=CC=CC=C5)C(P(C6=CC=CC=C6)C7=CC=CC=C7)=CC=C2",
    ),
    (  # dim 4: solvent -- 4 options (MeCN, THF, DMF, MeOH)
        "N#CC",
        "C1COCC1",
        "O=CN(C)C",
        "CO",
    ),
)

D = len(PARAM_OPTIONS)
N_OPTS_PER_DIM: Tuple[int, ...] = tuple(len(opts) for opts in PARAM_OPTIONS)
TOTAL_COMBOS = int(np.prod(N_OPTS_PER_DIM))  # 3,696 reactions

SOLVENT_DIM = 4  # HET splits agents by solvent.

# Full reaction grid; each agent gets either all rows or one solvent slice.
ALL_CANDIDATES_INT = np.array(
    list(product(*[range(n) for n in N_OPTS_PER_DIM])), dtype=np.int64
)


def decode_int_to_smiles(x_int: Sequence[int]) -> List[str]:
    """Convert category IDs to the corresponding SMILES strings."""
    return [PARAM_OPTIONS[d][int(x_int[d])] for d in range(D)]


def build_lookup_table() -> Dict[Tuple[int, ...], float]:
    """Load ``suzuki_edbo`` as ``{category_tuple: mean_yield}``.

    Yields are percentages in ``[0, 100]``. If the raw dataset contains repeated
    reaction conditions, their yields are averaged to match the reference.
    """
    from olympus.datasets.dataset import load_dataset

    data, ds_cfg, _, _, _ = load_dataset("suzuki_edbo")
    meas_names = [m["name"] for m in ds_cfg["measurements"]]
    meas_idx = meas_names.index("yield") if "yield" in meas_names else 0
    n_params = len(ds_cfg["parameters"])

    smiles_to_idx = [{opts[i]: i for i in range(len(opts))} for opts in PARAM_OPTIONS]

    agg: Dict[Tuple[int, ...], List[float]] = {}
    for row in data:
        smiles_key = tuple(str(v) for v in row[:n_params])
        try:
            int_key = tuple(smiles_to_idx[d][smiles_key[d]] for d in range(D))
        except KeyError as exc:  # a SMILES the option table does not know about
            raise RuntimeError(
                f"row {smiles_key} contains a SMILES not in PARAM_OPTIONS: {exc}"
            ) from exc
        agg.setdefault(int_key, []).append(float(row[n_params + meas_idx]))

    return {k: float(np.mean(v)) for k, v in agg.items()}


def make_evaluator(lookup: Dict[Tuple[int, ...], float]):
    """Return the local objective ``design point -> yield``.

    This is ADKO Algorithm 1 step 9: the only time an agent touches ground truth.
    A missing key is an error because it means the candidate space and dataset no
    longer match.
    """

    def evaluate(point: Sequence[int]) -> float:
        key = tuple(int(v) for v in point)
        if key not in lookup:
            raise RuntimeError(f"oracle lookup miss: {key}")
        return lookup[key]

    return evaluate


class SuzukiSpace(DesignSpace):
    """One agent's allowed Suzuki reaction space.

    In IID mode, every agent can choose any of the 3,696 reactions. In HET mode,
    each agent is restricted to one solvent, giving four disjoint 924-point
    slices. That makes token sharing the only way to benefit from another
    solvent's observations.

    ``embed`` is intentionally the identity map on category IDs. Token-location
    noise is applied only when a token is emitted, not when fitting the GP.
    """

    def __init__(self, agent_index: int, iid_mode: bool, seed: int = 0):
        self.agent_index = int(agent_index)
        self.iid_mode = bool(iid_mode)
        self.space_id = f"suzuki_edbo:{'iid' if iid_mode else 'het'}"
        self._rng = np.random.default_rng(seed)

        if iid_mode:
            allowed = ALL_CANDIDATES_INT
        else:
            allowed = ALL_CANDIDATES_INT[
                ALL_CANDIDATES_INT[:, SOLVENT_DIM] == self.agent_index
            ]
        # Store once because `enumerate()` is called every round.
        self._allowed: List[Tuple[int, ...]] = [
            tuple(int(v) for v in row) for row in allowed
        ]

    def embed(self, point: Sequence[int]) -> List[float]:
        return [float(v) for v in point]

    def enumerate(self) -> List[Tuple[int, ...]]:
        return self._allowed

    def sample(self, n: int, seed: Optional[int] = None) -> List[Tuple[int, ...]]:
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        idx = rng.integers(0, len(self._allowed), size=n)
        return [self._allowed[int(i)] for i in idx]


def flip_categories(
    point: Sequence[float], p_noise: float, rng: np.random.Generator
) -> List[float]:
    """Apply reference-style token-location noise.

    Each category is independently replaced by a different valid category with
    probability ``p_noise``. This is for outgoing token embeddings only; the
    agent's own GP keeps the true point.
    """
    noisy = [float(v) for v in point]
    if p_noise <= 0.0:
        return noisy
    for dim, n_opts in enumerate(N_OPTS_PER_DIM):
        if n_opts <= 1:
            continue
        if float(rng.random()) < p_noise:
            true_val = int(noisy[dim])
            alt = int(rng.integers(0, n_opts - 1))
            if alt >= true_val:  # force an actual category change
                alt += 1
            noisy[dim] = float(alt)
    return noisy
