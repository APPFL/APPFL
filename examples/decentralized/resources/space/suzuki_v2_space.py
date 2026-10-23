"""Suzuki-Miyaura search space for ADKO v2, and its yield table.

A design point is a tuple of five category ids, ``(electrophile, nucleophile, base, ligand,
solvent)``, over a grid of 4 x 3 x 7 x 11 x 4 = 3,696 measured reactions. Ported from the v2
reference implementation, so a run here matches a run there:

* the grid is in mixed-radix (C) order, which is the order the score's ``argmax`` breaks ties in;
* **HET** restricts lab ``i`` to solvent ``i mod 4`` (two labs per solvent with eight labs);
  **IID** opens every reaction to every lab;
* the privacy channel flips each category of an outgoing token to another option with
  probability 0.04365, drawn from the agent's own token stream;
* the warmup is the reference's shared warmup: lab ``i``'s points are drawn from the
  ``(i + 1)``-th seed a ``default_rng(seed)`` emits, so v1 and v2 runs pair seed for seed.

The oracle is the measured table, vendored as ``../evaluator/suzuki_edbo.csv`` (3,696 rows,
yields in [0, 100]) the way the reference vendors it: built from Olympus'
``dataset_suzuki_edbo/data.csv`` (MIT licence; Perera et al., *Science* 359, 429-434, 2018) at
Olympus commit ``f60267eb``, one row per condition in mixed-radix order. No Olympus needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from appfl.decentralized.algorithm.adko_v2 import DesignSpace

RADIX: Tuple[int, ...] = (4, 3, 7, 11, 4)
SOLVENT = 4
TOTAL = int(np.prod(RADIX))
#: Every reaction, in mixed-radix (C) order -- the row order of the table.
ALL = np.array(np.unravel_index(np.arange(TOTAL), RADIX), dtype=np.int64).T
#: Per-dimension probability that a token reports a different option.
P_NOISE = 0.04365

TABLE_PATH = Path(__file__).resolve().parents[1] / "evaluator" / "suzuki_edbo.csv"
_YIELD: Optional[np.ndarray] = None


def flat_index(x_int: Sequence[int]):
    """Table row of a condition, shape ``(5,)`` or ``(n, 5)``."""
    return np.ravel_multi_index(np.asarray(x_int, dtype=np.int64).T, RADIX)


def table() -> np.ndarray:
    """Yields in [0, 100], indexed by :func:`flat_index`. Loaded once per process."""
    global _YIELD
    if _YIELD is None:
        data = np.loadtxt(TABLE_PATH, delimiter=",", skiprows=1)
        if not np.array_equal(flat_index(data[:, :5]), np.arange(len(data))):
            raise ValueError(f"{TABLE_PATH} is not in mixed-radix order")
        _YIELD = data[:, 5]
        _YIELD.setflags(write=False)
    return _YIELD


def make_evaluator() -> Callable[[Sequence[int]], float]:
    """The local objective ``condition -> yield``: ADKO Algorithm 1 step 9."""

    def evaluate(point: Sequence[int]) -> float:
        return float(table()[flat_index(point)])

    return evaluate


def solvent_of(agent: int) -> int:
    """A HET lab's solvent: its index mod 4."""
    return int(agent) % RADIX[SOLVENT]


def flip_categorical(
    x_int: Sequence[float], n_opts: Sequence[int], p: float, rng
) -> np.ndarray:
    """Each dim moves to another option with probability ``p`` (the reference's channel)."""
    x = np.array(x_int, copy=True)
    for dim, n in enumerate(n_opts):
        if n <= 1:
            continue
        if float(rng.random()) < p:
            alt = int(rng.integers(0, n - 1))
            x[dim] = alt + 1 if alt >= int(x_int[dim]) else alt
    return x


class SuzukiSpace(DesignSpace):
    """One lab's Suzuki slice: all 3,696 reactions (IID) or its solvent's 924 (HET).

    ``embed`` is the identity on category ids: the GP reads them through a categorical kernel,
    and the peer field compares them under Hamming distance.
    """

    def __init__(
        self, agent_index: int = 0, iid_mode: bool = False, p_noise: float = P_NOISE
    ):
        self.agent_index = int(agent_index)
        self.iid_mode = bool(iid_mode)
        self.setting = "iid" if self.iid_mode else "het"
        self.p_noise = float(p_noise)
        # The reference's own array: `ALL` itself in IID -- a transposed, column-major view --
        # and a row-major copy of the solvent's rows in HET. `grid()` hands it to the agent as
        # is, so the peer field's matrix products sum in the reference's order.
        self._rows = (
            ALL
            if self.iid_mode
            else ALL[ALL[:, SOLVENT] == solvent_of(self.agent_index)]
        )
        self.space_id = (
            "suzuki_edbo:iid"
            if self.iid_mode
            else f"suzuki_edbo:het:solvent{solvent_of(self.agent_index)}"
        )
        self._points: List[Tuple[int, ...]] = [
            tuple(int(v) for v in row) for row in self._rows
        ]

    def enumerate(self) -> List[Tuple[int, ...]]:
        return self._points

    def grid(self) -> np.ndarray:
        return np.asarray(self._rows, dtype=float)

    def embed(self, point: Sequence[int]) -> List[float]:
        return [float(v) for v in point]

    def privatize(
        self, embedding: Sequence[float], rng: np.random.Generator
    ) -> List[int]:
        return [int(v) for v in flip_categorical(embedding, RADIX, self.p_noise, rng)]

    def warmup_points(self, seed: int, rounds: int) -> List[Tuple[int, ...]]:
        """The reference's shared warmup (v1's ``_generate_base_warmup_payload`` scheme)."""
        master = np.random.default_rng(seed)
        draws = [int(master.integers(0, 2**31)) for _ in range(self.agent_index + 1)]
        rng = np.random.default_rng(draws[-1])
        points = []
        for _ in range(int(rounds)):
            if self.iid_mode:
                x = [rng.integers(0, n) for n in RADIX]
            else:
                x = [rng.integers(0, n) for n in RADIX[:SOLVENT]]
                x.append(solvent_of(self.agent_index))
            points.append(tuple(int(v) for v in x))
        return points

    def allowed(self) -> List[List[int]]:
        return [
            list(range(n))
            if self.iid_mode or d != SOLVENT
            else [solvent_of(self.agent_index)]
            for d, n in enumerate(RADIX)
        ]

    def llm_context(self) -> Dict[str, Any]:
        return {"setting": self.setting}


def get_design_space(
    agent_index: int = 0,
    num_agents: int = 4,
    iid_mode: bool = False,
    p_noise: float = P_NOISE,
) -> SuzukiSpace:
    """Entry point named by ``space_configs.space_name``.

    ``iid_mode`` is the heterogeneity switch. Unlike the v1 example, HET runs with any number
    of labs: lab ``i`` gets solvent ``i mod 4``, as in the reference's 8-lab runs.
    """
    if int(num_agents) < 2:
        raise ValueError("ADKO needs at least two labs")
    return SuzukiSpace(agent_index, iid_mode, p_noise)
