"""The communication budget -- a cap on what one agent may send to one neighbour per round.

Enforcement, not observation: tightening this changes what crosses the wire and therefore what
the federation learns. Counting what actually crossed is ``metrics.py``, which is deliberately
separate -- one is part of the system, the other watches it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from appfl.decentralized.protocol import TokenProtocol


@dataclass
class CommBudget:
    """A cap on what one agent may send to one neighbour in one round.

    Two independent limits: how many tokens, and how many bits. ``None`` means unlimited --
    the right setting for full-communication baseline arms.

    This is the generic shape of a bandwidth constraint; ADKO's Constraint 3.2 is one
    instance of it, fixing ``tokens_per_neighbor_per_round = 1``. The reason it matters there:
    ADKO reports 333 B/round against 232.7 MB/round for FedAvg BO, roughly six orders of
    magnitude, and that gap is the argument for the whole approach at DOE scale.
    """

    bits_per_neighbor_per_round: Optional[int] = None
    tokens_per_neighbor_per_round: int = 1  # "at most one token" in the paper

    def admits(self, tokens: Sequence[TokenProtocol]) -> List[TokenProtocol]:
        """Trim an outgoing batch to what the budget permits for a single neighbour."""
        allowed = list(tokens)[: self.tokens_per_neighbor_per_round]
        if self.bits_per_neighbor_per_round is None:
            return allowed
        kept, spent = [], 0
        for token in allowed:
            size = token.size_bits()
            if spent + size > self.bits_per_neighbor_per_round:
                continue
            kept.append(token)
            spent += size
        return kept
