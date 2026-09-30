"""MPI simulation of a decentralized federation -- one agent per rank, peer-to-peer.

The HPC path. Same agents, same algorithm, same numbers as the serial run; the only difference
is that tokens cross ranks instead of dict entries. This is what scales to agent counts a
single process cannot reach, and therefore what a coordination-scaling study runs on.

    mpirun -n 4 python mpi/run_mpi.py

Rank r owns agent r, so -n must equal ``federation_configs.num_agents``. Note that there is no
rank 0 coordinator: rank 0 owns an agent like every other rank and only does the final gather
so that one process can print the result block.
"""

import argparse
import sys
from pathlib import Path

from mpi4py import MPI

#: ``examples/decentralized``. Relative resource paths in the configs resolve against it, so
#: a launcher can be invoked from here or from the repository root.
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_ROOT / "resources"))

from appfl.decentralized import (
    MPIExchange,
    create_budget,
    create_topology,
    load_agent_configs,
    load_federation_config,
    run_local_agent,
)
from appfl.decentralized.algorithm.adko import ADKOMeter, create_agent

from llm_cli import add_llm_arguments, llm_config_from_args  # noqa: E402
from reporting import describe_slice, format_point, print_config, print_llm, print_totals  # noqa: E402


def main() -> None:
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument(
        "--federation_config", type=str, default="./resources/configs/toy1d/federation_adko.yaml"
    )
    argparser.add_argument(
        "--agent_config", type=str, default="./resources/configs/toy1d/agent.yaml"
    )
    argparser.add_argument("--num_rounds", type=int, default=None)
    add_llm_arguments(argparser)
    args = argparser.parse_args()

    comm = MPI.COMM_WORLD
    # No --num_agents here: the agent count is -n, and a flag that could disagree with it
    # would just be a second place to get it wrong.
    federation_config = load_federation_config(args.federation_config, EXAMPLE_ROOT)
    if args.num_rounds is not None:
        federation_config.federation_configs.num_rounds = args.num_rounds
    federation = federation_config.federation_configs
    n_rounds = int(federation.num_rounds)
    if comm.Get_size() != int(federation.num_agents):
        raise SystemExit(
            f"-n {comm.Get_size()} but the config asks for {federation.num_agents} agents; "
            f"this launcher owns exactly one agent per rank"
        )

    topology = create_topology(federation_config)
    meter = ADKOMeter()
    exchange = MPIExchange(
        topology, comm=comm, budget=create_budget(federation_config), meter=meter
    )
    agent_configs = load_agent_configs(args.agent_config, federation_config, EXAMPLE_ROOT)
    agent_config = agent_configs[comm.Get_rank()]
    agent = create_agent(
        agent_config, federation_config, topology, meter, llm_config_from_args(args)
    )

    run_local_agent(
        agent, exchange, n_rounds, meter=meter,
        on_round_end=lambda _round, local: meter.record_fidelity([local]),
    )

    # Reduce onto rank 0 so the output block matches the serial run's.
    gathered = exchange.gather_results(
        {
            "agent_id": agent.agent_id,
            "slice": describe_slice(agent),
            "best": agent.best_so_far(),
            "eta_bar": agent.mean_token_fidelity(),
            "meter": meter,
            # stats(), not the model itself -- the client holds a socket and a SQLite
            # connection and does not survive pickling across ranks.
            "llm": (
                agent.language_model.stats() if agent.language_model is not None else None
            ),
        }
    )
    if comm.Get_rank() != 0:
        return

    total = ADKOMeter()
    for row in gathered:
        total.merge(row["meter"])
        if row["best"] is not None:
            total.best_by_round.append(row["best"][1])
    print(f"\n=== mpi, {federation.get('topology', 'fully_connected')} ===")
    print(f"topology            : {topology.describe()}")
    print_config([agent])
    for rank, row in enumerate(gathered):
        if row["best"] is not None:
            print(
                f"  {row['agent_id']} (rank {rank}) {row['slice']}  "
                f"best x={format_point(row['best'][0])} yield={row['best'][1]:.1f}  "
                f"eta_bar={row['eta_bar']:.3f}"
            )
    print_totals(total, n_rounds, optimum=federation.get("known_optimum", None))
    print_llm([row["llm"] for row in gathered])
    print(
        "\nCompare against serial/run_serial.py with the same configs: the numbers should"
        "\nmatch. If they don't, distribution changed the algorithm, not just its"
        "\nplumbing -- which is exactly the bug this pair of launchers exists to catch."
    )


if __name__ == "__main__":
    main()
