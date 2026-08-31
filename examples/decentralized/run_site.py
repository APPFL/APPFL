"""One site, one agent -- run this at each participating institution.

Its data, its surrogate, and its raw observations never leave this process. The only thing
that goes on the wire is one knowledge token per round.

    # in four separate terminals, or at four separate institutions
    python examples/decentralized/run_site.py --agent-id agent-0 --server-uri localhost:50051
    python examples/decentralized/run_site.py --agent-id agent-1 --server-uri localhost:50051
    python examples/decentralized/run_site.py --agent-id agent-2 --server-uri localhost:50051
    python examples/decentralized/run_site.py --agent-id agent-3 --server-uri localhost:50051

Sites may start in any order; the round barrier holds until all four have arrived. Add
--use-ssl --root-certificate for a real network.
"""

import argparse

from appfl.comm.grpc import GRPCClientCommunicator

from appfl.decentralized import CommBudget, RelayExchange, run_local_agent
from appfl.decentralized.algorithm.adko import ADKOMeter

from llm_cli import add_llm_arguments, llm_config_from_args
from benchmark import AGENT_IDS, make_agent, make_topology


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-id", required=True, choices=AGENT_IDS)
    parser.add_argument("--server-uri", default="localhost:50051")
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
    parser.add_argument("--use-ssl", action="store_true")
    parser.add_argument("--root-certificate", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-rounds", type=int, default=5,
                        help="random proposals before the surrogate is trusted")
    add_llm_arguments(parser)
    args = parser.parse_args()
    llm_config = llm_config_from_args(args)

    # The topology must match the relay's, or an agent will weight peers it isn't actually
    # connected to. In a real deployment this comes from a shared config, not a flag.
    topology = make_topology(args.topology)

    communicator = GRPCClientCommunicator(
        client_id=args.agent_id,
        server_uri=args.server_uri,
        use_ssl=args.use_ssl,
        root_certificate=args.root_certificate,
    )

    meter = ADKOMeter()
    exchange = RelayExchange(
        topology,
        agent_id=args.agent_id,
        communicator=communicator,
        budget=CommBudget(bits_per_neighbor_per_round=args.bits_per_neighbor),
        meter=meter,
    )
    agent = make_agent(
        args.agent_id, topology, meter,
        preset=args.preset, lam=args.lam, gamma=args.gamma,
        token_budget=args.token_budget, seed=args.seed,
        warmup_rounds=args.warmup_rounds,
        llm_config=llm_config,
    )

    def trace(round_idx, agent):
        meter.record_fidelity([agent])
        best = agent.best_so_far()
        if best is not None and (round_idx + 1) % 10 == 0:
            print(
                f"[{args.agent_id}] round {round_idx + 1:>3}  "
                f"best x={best[0]:.3f} yield={best[1]:.1f}  "
                f"eta_bar={agent.mean_token_fidelity():.3f}  "
                f"bits sent={meter.bits_sent}"
            )

    print(f"[{args.agent_id}] joining {args.server_uri}, "
          f"neighbors {topology.neighbors(args.agent_id)}")
    run_local_agent(agent, exchange, args.rounds, meter=meter, on_round_end=trace)

    best = agent.best_so_far()
    print(f"\n[{args.agent_id}] done after {args.rounds} rounds")
    if best is not None:
        print(f"  best found        : x={best[0]:.3f} yield={best[1]:.1f}")
    print(f"  eta_bar           : {agent.mean_token_fidelity():.3f}")
    print(f"  tokens emitted    : {meter.tokens_emitted}")
    print(f"  bits sent         : {meter.bits_sent}")
    print(f"  bits per round    : {meter.bits_per_round(args.rounds):.0f}")
    print(f"  raw observations shared: 0  (Constraint 3.1)")


if __name__ == "__main__":
    main()
