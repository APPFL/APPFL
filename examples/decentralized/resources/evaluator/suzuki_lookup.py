"""The Suzuki oracle -- ADKO Algorithm 1 step 9 against the real ``suzuki_edbo`` table.

Evaluating a point means looking up its measured reaction yield, not calling a synthetic
objective. The table is the same one the reference ADKO implementation uses.

Requires Olympus (``pip install olymp``); see ``reference/README.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

_SPACE_DIR = Path(__file__).resolve().parents[1] / "space"
if str(_SPACE_DIR) not in sys.path:
    sys.path.insert(0, str(_SPACE_DIR))

from suzuki_space import build_lookup_table, make_evaluator  # noqa: E402

#: Loading the Olympus dataset takes seconds and every agent in a serial run asks for the
#: same table, so it is built once per process.
_LOOKUP: Optional[Dict[Tuple[int, ...], float]] = None


def get_evaluator() -> Callable[[Sequence[int]], float]:
    """Entry point named by ``evaluator_configs.evaluator_name``."""
    global _LOOKUP
    if _LOOKUP is None:
        _LOOKUP = build_lookup_table()
    return make_evaluator(_LOOKUP)
