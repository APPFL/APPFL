"""Run metering -- what crossed the wire, and what it cost.

Observation, not enforcement: nothing here changes how a run behaves. Delete it and the
federation produces byte-identical results, you simply cannot see them. The constraint that
*does* change behaviour lives in ``budget.py``.

Every field is something any token-exchanging method produces, so the same record comes out of
an in-process run, an ``mpirun`` run, and a live multi-site run. That comparability is the
point: it is what lets a scaling curve measured in simulation be read against a demo measured
across real institutions. Algorithm-specific traces subclass :class:`Meter` -- see
``adko.metrics.ADKOMeter``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Sequence

from appfl.decentralized.protocol import TokenProtocol


def metered_evaluator(
    evaluator: Callable[[Any], float],
    meter: "Meter",
    cost_seconds: float = 0.0,
) -> Callable[[Any], float]:
    """Wrap an agent's evaluation callable so every call is counted.

    The agent still just calls ``self.evaluator(point)``; the accounting happens around it,
    which is also where compute-budget awareness would attach without the agent changing.
    ``cost_seconds`` overrides measured wall time, for benchmarks where the stand-in
    evaluation is instant but should be priced as if it were a real experiment.
    """

    def wrapped(point: Any) -> float:
        import time

        start = time.time()
        value = evaluator(point)
        meter.compute_seconds += cost_seconds or (time.time() - start)
        meter.evaluations += 1
        return value

    return wrapped


@dataclass
class Meter:
    """What a run reports. Transport-independent on purpose.

    Because every backend records through the same object, a scaling curve measured under
    MPI can be read against a demo measured across real sites -- which is the whole reason
    for having one agent implementation and three transports.
    """

    bits_sent: int = 0
    messages_sent: int = 0
    tokens_emitted: int = 0
    tokens_dropped_budget: int = 0
    evaluations: int = 0
    compute_seconds: float = 0.0
    best_by_round: List[float] = field(default_factory=list)

    def record_sent(self, tokens: Sequence[TokenProtocol]) -> None:
        """Count one delivery of ``tokens`` to one neighbor."""
        for token in tokens:
            self.bits_sent += token.size_bits()
            self.messages_sent += 1

    def record_emitted(self, tokens: Sequence[TokenProtocol]) -> None:
        self.tokens_emitted += len(tokens)

    def bits_per_round(self, n_rounds: int) -> float:
        """Traffic per round. Compare against ADKO's published 333 B (2664 bits)."""
        return 0.0 if n_rounds == 0 else self.bits_sent / n_rounds

    def bits_per_evaluation(self) -> float:
        """Communication efficiency: the x-axis of a coordination-scaling curve."""
        return 0.0 if self.evaluations == 0 else self.bits_sent / self.evaluations

    def merge(self, other: "Meter") -> None:
        """Fold another meter in. Used to reduce per-rank meters onto rank 0 after an MPI run."""
        self.bits_sent += other.bits_sent
        self.messages_sent += other.messages_sent
        self.tokens_emitted += other.tokens_emitted
        self.tokens_dropped_budget += other.tokens_dropped_budget
        self.evaluations += other.evaluations
        self.compute_seconds += other.compute_seconds
