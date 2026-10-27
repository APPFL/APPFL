"""The oracle for the toy experiment -- ADKO Algorithm 1 step 9, the local measurement.

The decentralized analogue of ``examples/resources/loss``: the one function that touches
ground truth. It is loaded per agent, because in a real federation each site has its own
apparatus; here all four happen to call the same closed form.
"""

from __future__ import annotations

import math
from typing import Callable

#: ADKO's tau and the advantage-score normalizer, on the 0-100 yield scale.
THRESHOLD = 50.0
SCALE = 50.0


def true_yield(x: float) -> float:
    """Stand-in for an expensive evaluation: a percentage yield in [0, 100].

    Peak ~100 at x=0.35, decoy ~40 at x=0.75. Only the windows near 0.35 can reach the peak
    alone, so the remaining agents can only find it by acting on what a neighbor tells them.
    """
    return 100.0 * math.exp(-((x - 0.35) ** 2) / 0.01) + 40.0 * math.exp(
        -((x - 0.75) ** 2) / 0.02
    )


def get_evaluator() -> Callable[[float], float]:
    """Entry point named by ``evaluator_configs.evaluator_name``.

    The launcher wraps whatever comes back in :func:`appfl.decentralized.metered_evaluator`,
    so evaluation counts and simulated cost are recorded without this file knowing.
    """
    return true_yield
