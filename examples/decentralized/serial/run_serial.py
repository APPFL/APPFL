"""Serial simulation of a decentralized federation -- every agent in one process.

The control run. No transport, no ranks, no network: tokens move between dict entries. Use it
to get a result quickly, and as the reference output the MPI and gRPC runs must match.

    python serial/run_serial.py
    python serial/run_serial.py \
        --federation_config ./resources/configs/toy1d/federation_adko_tuned.yaml

Which experiment runs is entirely a matter of which configs are passed; see
./resources/configs.
"""

import argparse
import sys
from pathlib import Path

#: ``examples/decentralized``. Relative resource paths in the configs resolve against it, so
#: a launcher can be invoked from here or from the repository root.
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_ROOT / "resources"))

from appfl.decentralized import (
    InProcessExchange,
    create_budget,
    create_topology,
    load_agent_configs,
    load_federation_config,
    run_federation,
)
from appfl.decentralized.algorithm.adko import ADKOMeter, create_agent

from llm_cli import add_llm_arguments, llm_config_from_args  # noqa: E402
from reporting import report  # noqa: E402


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
    argparser.add_argument("--num_agents", type=int, default=None)
    argparser.add_argument("--num_rounds", type=int, default=None)
    add_llm_arguments(argparser)
    args = argparser.parse_args()

    # Load the federation configuration and override the two fields worth a flag.
    federation_config = load_federation_config(args.federation_config, EXAMPLE_ROOT)
    if args.num_agents is not None:
        federation_config.federation_configs.num_agents = args.num_agents
    if args.num_rounds is not None:
        federation_config.federation_configs.num_rounds = args.num_rounds
    federation = federation_config.federation_configs
    n_rounds = int(federation.num_rounds)

    topology = create_topology(federation_config)
    meter = ADKOMeter()
    exchange = InProcessExchange(
        topology, budget=create_budget(federation_config), meter=meter
    )

    llm_config = llm_config_from_args(args)
    agents = [
        create_agent(agent_config, federation_config, topology, meter, llm_config)
        for agent_config in load_agent_configs(args.agent_config, federation_config, EXAMPLE_ROOT)
    ]

    # eta_bar is ADKO's own trace, so it attaches through the driver's hook rather than the
    # driver knowing about it.
    run_federation(
        agents, exchange, n_rounds, meter=meter,
        on_round_end=lambda _round, round_agents: meter.record_fidelity(round_agents),
    )
    report(
        f"serial, {federation.get('topology', 'fully_connected')}",
        topology, agents, meter, n_rounds,
        optimum=federation.get("known_optimum", None),
    )


if __name__ == "__main__":
    main()
