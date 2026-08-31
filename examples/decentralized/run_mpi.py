"""ADKO across MPI ranks -- one agent per rank, peer-to-peer, no coordinator.

The HPC path. Same agents, same algorithm, same numbers as the in-process run; the only
difference is that tokens cross ranks instead of dict entries. This is what scales to agent
counts an in-process demo cannot reach, and therefore what a coordination-scaling study runs
on.

    mpirun -n 4 python examples/decentralized/run_mpi.py

Rank r owns AGENT_IDS[r], so -n must equal the agent count.
"""

import argparse

from mpi4py import MPI

from appfl.decentralized import CommBudget, MPIExchange, run_local_agent
from appfl.decentralized.algorithm.adko import ADKOMeter

from llm_cli import add_llm_arguments, llm_config_from_args
from benchmark import AGENT_IDS, make_agent, make_topology, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--topology", default="fully_connected")
    parser.add_argument("--token-budget", type=int, default=40)
    parser.add_argument("--preset", default="many_task",
                        choices=["many_task", "suzuki"],
                        help="published configuration to follow "
                             "(baseline, weights, kernel, pruner)")
    parser.add_argument("--lam", type=float, default=None,
                        help="override the preset's attraction weight")
    parser.add_argument("--gamma", type=float, default=None,
                        help="override the preset's avoidance weight")
    parser.add_argument("--bits-per-neighbor", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-rounds", type=int, default=5,
                        help="random proposals before the surrogate is trusted")
    add_llm_arguments(parser)
    args = parser.parse_args()
    llm_config = llm_config_from_args(args)

    comm = MPI.COMM_WORLD
    topology = make_topology(args.topology)
    if comm.Get_rank() == 0:
        assert topology.fiedler_value() > 1e-9, "topology is disconnected"

    meter = ADKOMeter()
    exchange = MPIExchange(
        topology,
        comm=comm,
        budget=CommBudget(bits_per_neighbor_per_round=args.bits_per_neighbor),
        meter=meter,
    )
    agent = make_agent(
        exchange.agent_id, topology, meter,
        preset=args.preset, lam=args.lam, gamma=args.gamma,
        token_budget=args.token_budget, seed=args.seed,
        warmup_rounds=args.warmup_rounds,
        llm_config=llm_config,
    )

    run_local_agent(
        agent, exchange, args.rounds, meter=meter,
        on_round_end=lambda _round, a: meter.record_fidelity([a]),
    )

    # Reduce onto rank 0 so the output block matches the in-process run's.
    gathered = exchange.gather_results(
        {
            "agent_id": agent.agent_id,
            "best": agent.best_so_far(),
            "eta_bar": agent.mean_token_fidelity(),
            "meter": meter,
            # stats(), not the model itself -- the client holds a socket and a SQLite
            # connection and does not survive pickling across ranks.
            "llm": (
                agent.language_model.stats()
                if agent.language_model is not None
                else None
            ),
        }
    )
    if comm.Get_rank() == 0:
        total = ADKOMeter()
        for row in gathered:
            total.merge(row["meter"])
            if row["best"] is not None:
                total.best_by_round.append(row["best"][1])
        print(f"\n=== MPI, {args.topology}, {comm.Get_size()} ranks ===")
        print(f"topology            : {topology.describe()}")
        for rank, row in enumerate(gathered):
            if row["best"] is not None:
                print(
                    f"  {row['agent_id']} (rank {rank})  "
                    f"best x={row['best'][0]:.3f} yield={row['best'][1]:.1f}  "
                    f"eta_bar={row['eta_bar']:.3f}"
                )
        print(f"federation best     : {max(total.best_by_round):.1f}  (true optimum 100.0)")
        print(f"tokens emitted      : {total.tokens_emitted}")
        print(f"bits sent           : {total.bits_sent}")
        print(f"bits per round      : {total.bits_per_round(args.rounds):.0f}")
        print(f"evaluations         : {total.evaluations}")
        llm_rows = [row["llm"] for row in gathered if row["llm"]]
        if llm_rows:
            agg = {k: sum(r[k] for r in llm_rows) for k in ("calls", "cache_hits", "failures")}
            print(
                f"llm calls           : {agg['calls']} "
                f"({agg['cache_hits']} cached, {agg['failures']} failed)"
            )
        print(
            "\nCompare against run_inprocess.py with the same flags: the numbers should"
            "\nmatch. If they don't, distribution changed the algorithm, not just its"
            "\nplumbing -- which is exactly the bug this pair of scripts exists to catch."
        )


if __name__ == "__main__":
    main()
