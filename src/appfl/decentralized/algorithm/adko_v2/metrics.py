"""ADKO v2 run traces: the things only this algorithm produces.

v1 traces ``eta_bar``, mean token fidelity; v2 tokens carry no fidelity. Its analogue is trust:
whether agents still listen to each other, round by round. Like v1's ``ADKOMeter`` this extends
:class:`~appfl.decentralized.metrics.Meter` rather than living in it, so the generic metering
stays honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Sequence

from appfl.decentralized.metrics import Meter


@dataclass
class ADKOv2Meter(Meter):
    """:class:`Meter` plus the per-round mean trust."""

    mean_trust_by_round: List[float] = field(default_factory=list)

    def record_trust(self, agents: Sequence[Any]) -> None:
        """Append this round's mean trust, over the agents that keep any.

        Agents without trust (the equal-weight variant), or without ``mean_trust`` at all,
        contribute nothing, so a mixed federation is not an error.
        """
        values = [
            v
            for v in (getattr(a, "mean_trust", lambda: None)() for a in agents)
            if v is not None
        ]
        if values:
            self.mean_trust_by_round.append(sum(values) / len(values))

    def merge(self, other: Meter) -> None:
        """Fold in another meter, carrying its trust trace.

        As with v1's ``eta_bar`` trace, an MPI reduction concatenates each rank's series
        rather than averaging them round by round.
        """
        super().merge(other)
        if isinstance(other, ADKOv2Meter) and other.mean_trust_by_round:
            self.mean_trust_by_round.extend(other.mean_trust_by_round)
