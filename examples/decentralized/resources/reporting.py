"""One result block, printed the same way by every launcher.

The point of the example set is that the three transports produce the same numbers, and that
claim is only checkable if the three print the same thing. ``diff`` is the test.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from appfl.decentralized import Meter, Topology


def describe_slice(agent: Any) -> str:
    """``owns [lo,hi]`` for a space that has bounds, the space id otherwise."""
    space = getattr(agent, "space", None)
    lo, hi = getattr(space, "lo", None), getattr(space, "hi", None)
    if lo is not None and hi is not None:
        return f"owns [{lo:.2f},{hi:.2f}]"
    return f"space {getattr(space, 'space_id', '?')}"


def format_point(x: Any) -> str:
    """toy1d points are floats; Suzuki points are tuples of categorical choices."""
    return f"{x:.3f}" if isinstance(x, (int, float)) else f"{x}"


def print_config(agents: Sequence[Any]) -> None:
    if not agents:
        return
    weights = agents[0].weights
    print(
        f"config              : beta={weights.beta} lam={weights.lam} "
        f"gamma={weights.gamma} sigma_s={weights.sigma_s:.3f} kernel={weights.kernel} "
        f"fidelity_weighted={weights.weight_by_fidelity} norm={weights.peer_normalization}"
    )
    print(f"baseline            : {type(agents[0].baseline).__name__}")


def print_agent(agent: Any, rank: Optional[int] = None) -> None:
    best = agent.best_so_far()
    if best is None:
        return
    where = f" (rank {rank})" if rank is not None else ""
    print(
        f"  {agent.agent_id}{where} {describe_slice(agent)}  "
        f"best x={format_point(best[0])} yield={best[1]:.1f}  "
        f"eta_bar={agent.mean_token_fidelity():.3f}"
    )


def print_totals(meter: Meter, n_rounds: int, optimum: Optional[float] = None) -> None:
    if meter.best_by_round:
        suffix = f"  (true optimum {optimum:.1f})" if optimum is not None else ""
        print(f"federation best     : {max(meter.best_by_round):.1f}{suffix}")
    print(f"tokens emitted      : {meter.tokens_emitted}")
    print(f"bits sent           : {meter.bits_sent}")
    print(f"bits per round      : {meter.bits_per_round(n_rounds):.0f}")
    print(f"evaluations         : {meter.evaluations}")


def print_llm(stats: Sequence[dict]) -> None:
    rows = [s for s in stats if s]
    if not rows:
        return
    total = {k: sum(r[k] for r in rows) for k in ("calls", "cache_hits", "failures")}
    print(
        f"llm calls           : {total['calls']} "
        f"({total['cache_hits']} cached, {total['failures']} failed)"
    )


def report(
    label: str,
    topology: Topology,
    agents: Sequence[Any],
    meter: Meter,
    n_rounds: int,
    optimum: Optional[float] = None,
) -> None:
    """The block a serial or MPI run ends with. Identical fields, identical order."""
    print(f"\n=== {label} ===")
    print(f"topology            : {topology.describe()}")
    print_config(agents)
    for agent in agents:
        print_agent(agent)
    print_totals(meter, n_rounds, optimum)
    trace = getattr(meter, "mean_fidelity_by_round", None)
    if trace:
        print(f"eta_bar (last round): {trace[-1]:.3f}")
    print_llm(
        [
            a.language_model.stats()
            for a in agents
            if getattr(a, "language_model", None) is not None
        ]
    )
