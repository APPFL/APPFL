"""ADKO in one process -- the control run.

    python examples/decentralized/run_inprocess.py
"""

import argparse

from appfl.decentralized import CommBudget, InProcessExchange, run_federation
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
    parser.add_argument("--pruner", default=None,
                        choices=["fidelity", "confidence", "fifo", "random"],
                        help="override the preset's pruning rule")
    parser.add_argument("--bits-per-neighbor", type=int, default=None,
                        help="ADKO Constraint 3.2 budget; omit for unlimited")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-rounds", type=int, default=5,
                        help="random proposals before the surrogate is trusted")
    add_llm_arguments(parser)
    args = parser.parse_args()
    llm_config = llm_config_from_args(args)

    topology = make_topology(args.topology)
    assert topology.fiedler_value() > 1e-9, (
        "topology is disconnected; ADKO assumes a connected graph and its guarantees "
        "do not apply"
    )

    meter = ADKOMeter()
    exchange = InProcessExchange(
        topology,
        budget=CommBudget(bits_per_neighbor_per_round=args.bits_per_neighbor),
        meter=meter,
    )
    agents = [
        make_agent(
            agent_id, topology, meter,
            preset=args.preset, lam=args.lam, gamma=args.gamma,
            token_budget=args.token_budget, pruner_name=args.pruner, seed=args.seed,
            warmup_rounds=args.warmup_rounds,
            llm_config=llm_config,
        )
        for agent_id in AGENT_IDS
    ]

    # eta_bar is ADKO's own trace, so it attaches through the driver's hook rather
    # than the driver knowing about it.
    run_federation(
        agents, exchange, args.rounds, meter=meter,
        on_round_end=lambda _round, agents: meter.record_fidelity(agents),
    )
    report(f"in-process, {args.topology}", topology, agents, meter, args.rounds)


if __name__ == "__main__":
    main()
