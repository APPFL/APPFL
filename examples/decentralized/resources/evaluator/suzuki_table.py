"""The Suzuki oracle for ADKO v2 -- a lookup in the vendored ``suzuki_edbo.csv`` beside this file.

The same table the v2 reference implementation reads, parsed the
same way, so a yield here is the same float there. Needs no Olympus; the v1 example's
``suzuki_lookup.py`` reads the same data through it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Sequence

_SPACE_DIR = Path(__file__).resolve().parents[1] / "space"
if str(_SPACE_DIR) not in sys.path:
    sys.path.insert(0, str(_SPACE_DIR))

from suzuki_v2_space import make_evaluator  # noqa: E402


def get_evaluator() -> Callable[[Sequence[int]], float]:
    """Entry point named by ``evaluator_configs.evaluator_name``."""
    return make_evaluator()
