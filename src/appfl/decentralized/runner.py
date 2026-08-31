"""The round driver -- the single place where an algorithm meets a transport.

Everything backend-specific lives behind :class:`~appfl.decentralized.exchange.TokenExchange`,
so the loop is identical whether the federation is four objects in one process, 1024 MPI ranks
on a cluster, or four institutions on separate continents::

    collect  ->  agent.observe  ->  agent.act  ->  publish  ->  barrier

**A round is defined once, in :func:`take_turn`.** Both drivers call it and nothing else; the
only difference between them is how many agents this process is responsible for:

* :func:`run_federation` -- one process owns every agent. The in-process control run.
* :func:`run_local_agent` -- one process owns one agent, peers run elsewhere. An MPI rank or
  a remote site.

That is the whole difference. Delivery timing, budget enforcement and metering are identical
because they live in ``take_turn`` and in the transport, not in the loops. It matters because
the package's central claim is that the same experiment runs three ways and should produce the
same numbers -- and that claim is worthless if the in-process control and the MPI run go
through different loop bodies, since any disagreement could then be the driver rather than the
distribution.

Pairing a driver with the wrong transport raises rather than silently misbehaving; see
``TokenExchange.owns_single_agent``.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from appfl.decentralized.metrics import Meter
from appfl.decentralized.exchange import TokenExchange
from appfl.decentralized.protocol import AgentProtocol


def take_turn(
    agent: AgentProtocol,
    exchange: TokenExchange,
    meter: Meter,
    round_idx: int,
) -> None:
    """One agent's turn in one round. **The only place a round is defined.**

    Both drivers call this and nothing else, so "the same experiment three ways" is true by
    construction rather than by two loops being kept in step by hand. If this body is right,
    every backend is running the same algorithm; if it is wrong, they are all wrong the same
    way -- which is the property that makes cross-backend comparison mean anything.

    Note what is absent: any deferral of delivery. Published tokens do not reach a peer until
    ``exchange.barrier()``, and that is the transport's job in all three backends. Putting it
    here instead would make the driver's correctness depend on which transport was underneath.
    """
    agent.observe(exchange.collect(agent.agent_id))
    tokens = agent.act(round_idx)
    meter.record_emitted(tokens)
    exchange.publish(agent.agent_id, tokens)


def run_federation(
    agents: Sequence[AgentProtocol],
    exchange: TokenExchange,
    n_rounds: int,
    meter: Optional[Meter] = None,
    on_round_end: Optional[Callable[[int, Sequence[AgentProtocol]], None]] = None,
) -> Meter:
    """Drive every agent for ``n_rounds``. One process owns the whole federation.

    Each agent takes its turn, then the barrier delivers everything published during the
    round -- so no agent reads a token from the round it is still in. That matches the
    reference implementation, and it is enforced by the transport rather than by this loop.
    """
    if exchange.owns_single_agent:
        raise ValueError(
            f"{type(exchange).__name__} drives one agent per process; use run_local_agent. "
            f"run_federation needs a transport that holds every agent, such as "
            f"InProcessExchange."
        )
    meter = meter or exchange.meter
    for round_idx in range(n_rounds):
        for agent in agents:
            take_turn(agent, exchange, meter, round_idx)
        exchange.barrier(round_idx)
        _record_round(meter, agents)
        if on_round_end is not None:
            on_round_end(round_idx, agents)
    return meter


def run_local_agent(
    agent: AgentProtocol,
    exchange: TokenExchange,
    n_rounds: int,
    meter: Optional[Meter] = None,
    on_round_end: Optional[Callable[[int, AgentProtocol], None]] = None,
) -> Meter:
    """Drive one agent for ``n_rounds``; peers run elsewhere. MPI rank or remote site.

    Identical to :func:`run_federation` except that this process is responsible for one agent
    instead of all of them -- same :func:`take_turn`, same barrier.

    The barrier is what makes that equivalence hold: publish, then wait for every peer, so no
    agent starts round ``t+1`` reading a half-delivered round ``t``. Relaxing it -- letting a
    fast site run ahead on whatever arrived -- is the asynchronous variant, and that is a
    research question rather than a configuration flag, because ADKO's regret analysis
    assumes synchronous rounds.
    """
    if not exchange.owns_single_agent:
        raise ValueError(
            f"{type(exchange).__name__} holds every agent in one process; use "
            f"run_federation. run_local_agent would run this agent through all "
            f"{n_rounds} rounds before any peer started."
        )
    meter = meter or exchange.meter
    for round_idx in range(n_rounds):
        take_turn(agent, exchange, meter, round_idx)
        exchange.barrier(round_idx)
        _record_round(meter, [agent])
        if on_round_end is not None:
            on_round_end(round_idx, agent)
    return meter


def _record_round(meter: Meter, agents: Sequence[AgentProtocol]) -> None:
    """Per-round trace of what every agent has, whatever algorithm it runs.

    Only ``best_so_far``, which is on :class:`AgentProtocol` and so is available from any
    agent. Algorithm-specific traces attach through ``on_round_end`` instead -- ADKO's token
    fidelity goes through ``adko.metrics.ADKOMeter.record_fidelity``. Reaching into an agent
    for a method the protocol does not declare is how a generic driver quietly stops being
    generic.
    """
    observed = [b[1] for b in (a.best_so_far() for a in agents) if b is not None]
    if observed:
        meter.best_by_round.append(max(observed))
