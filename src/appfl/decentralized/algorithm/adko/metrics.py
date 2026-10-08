"""ADKO-specific run traces: the things only this algorithm produces.

``eta_bar`` -- mean token fidelity across an agent's memory -- is Proposition 4's quantity and
means nothing outside ADKO, so it extends :class:`~appfl.decentralized.metrics.Meter` rather
than living in it. That keeps the generic metering honest: every field on ``Meter`` is
something any token-exchanging method produces.

Wire it in through the round driver's existing ``on_round_end`` hook::

    meter = ADKOMeter()
    run_federation(agents, exchange, rounds, meter=meter,
                   on_round_end=lambda i, agents: meter.record_fidelity(agents))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Sequence

from appfl.decentralized.metrics import Meter


@dataclass
class ADKOMeter(Meter):
    """:class:`Meter` plus ADKO's per-round fidelity trace."""

    mean_fidelity_by_round: List[float] = field(default_factory=list)

    def record_fidelity(self, agents: Sequence[Any]) -> None:
        """Append this round's ``eta_bar``, averaged across agents.

        Worth watching every round rather than only at the end: if it drifts down as the
        token budget tightens, the compression term in the regret bound is not being
        controlled and sublinear convergence should not be expected. A run that looks fine
        on best-so-far can still be failing here.

        Agents without ``mean_token_fidelity`` contribute nothing, so a mixed federation --
        or a non-ADKO agent sharing the transport -- is not an error.
        """
        fidelities = [
            a.mean_token_fidelity()
            for a in agents
            if hasattr(a, "mean_token_fidelity")
        ]
        if fidelities:
            self.mean_fidelity_by_round.append(sum(fidelities) / len(fidelities))

    def merge(self, other: Meter) -> None:
        """Fold in another meter, carrying the fidelity trace when there is one.

        Overridden because :class:`Meter.merge` only knows its own fields; an added field
        that is not merged here is silently dropped in an MPI reduction.
        """
        super().merge(other)
        if isinstance(other, ADKOMeter) and other.mean_fidelity_by_round:
            # Per-rank traces are the same rounds measured by different agents, so extend
            # rather than sum -- rank 0 ends up with every agent's series, which is what the
            # report averages over.
            self.mean_fidelity_by_round.extend(other.mean_fidelity_by_round)
