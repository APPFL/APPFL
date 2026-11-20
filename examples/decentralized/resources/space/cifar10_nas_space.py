"""CIFAR-10 architecture search space -- the design space of the ADKO paper's CIFAR-10 study.

Mirrors ``cifar10/model.py`` in the reference implementation (github.com/lucasrillo/adko). A
design point is ``theta`` in ``[0, 1]^7``, decoded into a CIFAR-adapted ResNet and the recipe it
is trained with:

    dim 0  base_width        {8, 16, 32, 64}
    dim 1  blocks_per_group  {2, 3, 5}           (depth)
    dim 2  lr                10^(-4 + 3t)        [1e-4, 1e-1]
    dim 3  weight_decay      10^(-5 + 3t)        [1e-5, 1e-2]
    dim 4  dropout           0.5 t               [0, 0.5]
    dim 5  batch_size        {32, 64, 128, 256}
    dim 6  cosine schedule   t > 0.5

Every agent searches the same space. The heterogeneity in this study is in the *data* each
agent trains on, so it lives in the evaluator (``evaluator/cifar10_nas.py``) -- the reverse of
Suzuki, where the space is split by solvent and the oracle is shared.

The space is continuous, so ``enumerate`` returns ``None`` and the agent scores sampled
candidates instead of a grid. ``local_perturbations`` is left at the base-class default, a
uniform draw, which is exactly what the reference does: ``CIFARAgent.select`` scores
``n_candidates`` uniform points per round and never perturbs around the incumbent.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from appfl.decentralized.algorithm.adko import DesignSpace

BASE_WIDTH_CHOICES = (8, 16, 32, 64)
BLOCK_CHOICES = (2, 3, 5)
BATCH_CHOICES = (32, 64, 128, 256)
D_DESIGN = 7


def _pick(value: float, choices: Sequence[int]) -> int:
    return choices[min(int(value * len(choices)), len(choices) - 1)]


def decode_theta(theta: Sequence[float]) -> dict:
    """``theta`` in ``[0, 1]^7`` -> the network and training recipe it names."""
    t = np.clip(np.asarray(theta, dtype=float), 0.0, 1.0)
    return dict(
        base_width=_pick(t[0], BASE_WIDTH_CHOICES),
        blocks_per_group=_pick(t[1], BLOCK_CHOICES),
        lr=10 ** (-4 + t[2] * 3),
        wd=10 ** (-5 + t[3] * 3),
        dropout=t[4] * 0.5,
        batch_size=_pick(t[5], BATCH_CHOICES),
        cosine_schedule=bool(t[6] > 0.5),
    )


def readable_theta(theta: Sequence[float]) -> str:
    p = decode_theta(theta)
    sched = "cos" if p["cosine_schedule"] else "flat"
    return (
        f"w={p['base_width']}, blk={p['blocks_per_group']}, lr={p['lr']:.5f}, "
        f"wd={p['wd']:.6f}, drop={p['dropout']:.2f}, bs={p['batch_size']}, {sched}"
    )


class CIFAR10NASSpace(DesignSpace):
    """The unit hypercube ``[0, 1]^7``. ``embed`` is the identity -- the reference's
    ``sigma_phi = 0`` setting; see the config for why noise is not applied here."""

    space_id = "cifar10_nas"

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)

    def embed(self, point: Sequence[float]) -> list[float]:
        return [float(v) for v in point]

    def sample(self, n: int, seed: int | None = None) -> list[tuple[float, ...]]:
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        return [
            tuple(float(v) for v in row) for row in rng.uniform(0.0, 1.0, (n, D_DESIGN))
        ]


def get_design_space(
    agent_index: int = 0, num_agents: int = 5, seed: int = 0
) -> CIFAR10NASSpace:
    """Entry point named by ``space_configs.space_name``.

    ``agent_index`` and ``num_agents`` are filled in by the launchers. The space itself does not
    depend on them beyond the seed offset; the evaluator reads them to pick this agent's shard.
    """
    return CIFAR10NASSpace(seed=seed + agent_index)
