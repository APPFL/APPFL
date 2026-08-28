"""The baseline contract: how an agent decides what counts as SUCCESS."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple


class BaseBaseline(ABC):
    """Supplies ``(b, scale)`` for token encoding, and consumes local observations.

    ADKO's signal is ``s = SUCCESS iff y >= b``, and the paper is explicit that ``b`` may be
    **fixed or updated from the agent's local history**:

    * A **fixed** baseline is right when "success" has a natural domain meaning. The Suzuki
      study uses ``b = 50`` -- a 50% reaction yield is a threshold a chemist would recognize.
    * A **running median** is the recommended default otherwise. It adapts to each agent's own
      objective scale and removes the need to hand-pick a threshold, which matters when agents
      optimize genuinely different objectives whose ranges nobody knows in advance.

    The two published case studies make opposite calls, so this is a real per-study choice
    rather than a detail.

    Writing your own
    ----------------
    Two methods. ``observe`` is called once per round with the agent's new measurement;
    ``current`` returns the ``(baseline, scale)`` the next token is encoded against::

        from appfl.decentralized.adko.baseline import BaseBaseline

        class QuantileBaseline(BaseBaseline):
            \"\"\"Succeed only in the top quartile of what this agent has seen.\"\"\"

            def __init__(self, quantile: float = 0.75):
                self.quantile = quantile
                self.history = []

            def observe(self, observation: float) -> None:
                self.history.append(observation)

            def current(self):
                if not self.history:
                    return 0.0, 1.0
                ordered = sorted(self.history)
                index = min(len(ordered) - 1, int(self.quantile * len(ordered)))
                baseline = ordered[index]
                spread = max(ordered[-1] - ordered[0], 1e-8)
                return baseline, spread

    Then point a config at the file, exactly as APPFL does for a custom aggregator::

        baseline: QuantileBaseline
        baseline_path: ./my_baselines.py
        baseline_kwargs:
          quantile: 0.9

    Two properties any implementation should hold:

    * **``scale`` must be positive.** It divides the advantage score, and a zero makes every
      token meaningless rather than raising -- degenerate early histories are the usual cause,
      so fall back to something sensible rather than returning zero.
    * **Neither value is ever transmitted.** They are consumed inside ``encode_token`` and
      discarded; only the resulting sign and normalized magnitude leave the agent. An
      implementation that stashes history is fine, because that history stays local.
    """

    @abstractmethod
    def observe(self, observation: float) -> None:
        """Record one local observation. Never leaves the agent.

        Called before :meth:`current` in the same round, so an observation participates in
        its own baseline. That matches the reference implementation, which appends to the
        agent's history and then takes the median of it.
        """

    @abstractmethod
    def current(self) -> Tuple[float, float]:
        """Return ``(baseline, scale)`` to encode the next token with."""
