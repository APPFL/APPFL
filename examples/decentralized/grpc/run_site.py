"""One site, one agent -- run this at each participating institution.

Its design space, its surrogate, and its raw observations never leave this process. The only
thing that goes on the wire is one knowledge token per neighbour per round.

    # start the relay first, then in four separate terminals -- or at four institutions
    python grpc/run_site.py --agent_id agent-0
    python grpc/run_site.py --agent_id agent-1
    python grpc/run_site.py --agent_id agent-2
    python grpc/run_site.py --agent_id agent-3

Sites may start in any order; the round barrier holds until all of them have arrived. Each
site reads the same federation config -- the graph and the algorithm have to agree, or an
agent weights peers it is not actually connected to -- and its own agent config, which is the
only file that describes anything private. In a real deployment the endpoint and certificates
come from that agent config rather than from flags.
"""

import argparse
import sys
from pathlib import Path

#: ``examples/decentralized``. Relative resource paths in the configs resolve against it, so
#: a launcher can be invoked from here or from the repository root.
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_ROOT / "resources"))

from appfl.comm.grpc import GRPCClientCommunicator

from appfl.decentralized import (
    RelayExchange,
    agent_ids,
    create_budget,
    create_topology,
    load_agent_configs,
    load_federation_config,
    run_local_agent,
)
from appfl.decentralized.algorithm.adko import ADKOMeter, create_agent

from llm_cli import add_llm_arguments, llm_config_from_args  # noqa: E402
from reporting import describe_slice  # noqa: E402


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
    argparser.add_argument("--agent_id", type=str, required=True)
    argparser.add_argument("--num_rounds", type=int, default=None)
    # The endpoint belongs in the agent config; these are here for a local test run.
    argparser.add_argument("--server_uri", type=str, default=None)
    argparser.add_argument("--use_ssl", action="store_true")
    argparser.add_argument("--root_certificate", type=str, default=None)
    add_llm_arguments(argparser)
    args = argparser.parse_args()

    federation_config = load_federation_config(args.federation_config, EXAMPLE_ROOT)
    if args.num_rounds is not None:
        federation_config.federation_configs.num_rounds = args.num_rounds
    federation = federation_config.federation_configs
    n_rounds = int(federation.num_rounds)

    known = agent_ids(federation_config)
    if args.agent_id not in known:
        raise SystemExit(f"--agent_id {args.agent_id} is not one of {known}")
    agent_config = load_agent_configs(args.agent_config, federation_config, EXAMPLE_ROOT)[
        known.index(args.agent_id)
    ]

    grpc_configs = agent_config.get("comm_configs", {}).get("grpc_configs", {}) or {}
    server_uri = args.server_uri or grpc_configs.get("server_uri", "localhost:50051")
    communicator = GRPCClientCommunicator(
        client_id=args.agent_id,
        server_uri=server_uri,
        use_ssl=args.use_ssl or bool(grpc_configs.get("use_ssl", False)),
        root_certificate=args.root_certificate or grpc_configs.get("root_certificate", None),
    )

    topology = create_topology(federation_config)
    meter = ADKOMeter()
    exchange = RelayExchange(
        topology,
        agent_id=args.agent_id,
        communicator=communicator,
        budget=create_budget(federation_config),
        meter=meter,
    )
    agent = create_agent(
        agent_config, federation_config, topology, meter, llm_config_from_args(args)
    )

    def trace(round_idx, local):
        meter.record_fidelity([local])
        best = local.best_so_far()
        if best is not None and (round_idx + 1) % 10 == 0:
            print(
                f"[{args.agent_id}] round {round_idx + 1:>3}  "
                f"best x={best[0]:.3f} yield={best[1]:.1f}  "
                f"eta_bar={local.mean_token_fidelity():.3f}  "
                f"bits sent={meter.bits_sent}"
            )

    print(
        f"[{args.agent_id}] joining {server_uri}, {describe_slice(agent)}, "
        f"neighbors {topology.neighbors(args.agent_id)}"
    )
    run_local_agent(agent, exchange, n_rounds, meter=meter, on_round_end=trace)

    best = agent.best_so_far()
    print(f"\n[{args.agent_id}] done after {n_rounds} rounds")
    if best is not None:
        print(f"  best found        : x={best[0]:.3f} yield={best[1]:.1f}")
    print(f"  eta_bar           : {agent.mean_token_fidelity():.3f}")
    print(f"  tokens emitted    : {meter.tokens_emitted}")
    print(f"  bits sent         : {meter.bits_sent}")
    print(f"  bits per round    : {meter.bits_per_round(n_rounds):.0f}")
    print("  raw observations shared: 0  (Constraint 3.1)")


if __name__ == "__main__":
    main()
