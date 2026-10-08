"""Serial simulation of a decentralized optimization federation -- every node in one process.

The control run. No transport, no ranks, no network: tokens move between objects. Use it to
get a result quickly, and as the reference output the MPI and gRPC runs must match.

    # run from `examples`, like every other APPFL example
    python decentralized/serial/run_serial.py
    python decentralized/serial/run_serial.py --num_nodes 8 --num_rounds 10
    python decentralized/serial/run_serial.py \
        --config ./decentralized/resources/configs/toy1d/simulation/node_0.yaml

Which experiment runs is entirely a matter of which config is passed; see ./resources/configs.
Nodes 1..n-1 are derived from node 0's config by changing `node_id` and the agent index, the
same way the MNIST DFL examples derive theirs -- so one file configures the whole federation.
"""

import sys
import copy
import argparse
from omegaconf import OmegaConf
from appfl.agent import DeOptNodeAgent
from appfl.decentralized.algorithm.adko import create_agent
from appfl.decentralized.algorithm.adko.metrics import ADKOMeter
from appfl.decentralized.budget import CommBudget
from appfl.decentralized.neighbor import (
    build_topology_from_config,
    resolve_all_neighbors_from_topology,
)

sys.path.insert(0, "./decentralized/resources")

from llm_cli import add_llm_arguments, llm_config_from_args  # noqa: E402
from reporting import report  # noqa: E402

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--config",
    type=str,
    default="./decentralized/resources/configs/toy1d/simulation/node_0.yaml",
)
argparser.add_argument("--num_nodes", type=int, default=None)
argparser.add_argument("--num_rounds", type=int, default=None)
add_llm_arguments(argparser)
args = argparser.parse_args()

node_config = OmegaConf.load(args.config)
if args.num_nodes is not None:
    node_config.neighbors.num_nodes = args.num_nodes
    node_config.space_configs.space_kwargs.num_agents = args.num_nodes
if args.num_rounds is not None:
    node_config.num_epochs = args.num_rounds

topology = build_topology_from_config(node_config.neighbors)
node_ids = topology.node_ids
neighbors = resolve_all_neighbors_from_topology(topology, node_ids)
n_rounds = int(node_config.num_epochs)

print(f"{topology.describe()}")
for node_id in node_ids:
    print(f"  {node_id} receives from {list(neighbors[node_id].recv_from)}")

# One meter for the whole federation: in a serial run every node's traffic is this process's
# traffic, and the totals are what the three backends are compared on.
meter = ADKOMeter()
budget_configs = node_config.get("comm_configs", {}).get("budget_configs", {}) or {}
budget = CommBudget(
    bits_per_neighbor_per_round=budget_configs.get("bits_per_neighbor_per_round", None),
    tokens_per_neighbor_per_round=int(
        budget_configs.get("tokens_per_neighbor_per_round", 1)
    ),
)

nodes = []
for index, node_id in enumerate(node_ids):
    config = copy.deepcopy(node_config)
    config.node_id = node_id
    config.space_configs.space_kwargs.agent_index = index
    config.train_configs.logging_id = node_id
    algorithm_agent = create_agent(
        config,
        neighbors[node_id],
        meter,
        llm_config=llm_config_from_args(args),
        topology=topology,
    )
    nodes.append(
        DeOptNodeAgent(
            algorithm_agent=algorithm_agent,
            dfl_node_agent_config=config,
            neighbors=neighbors[node_id],
            budget=budget,
            meter=meter,
        )
    )

by_id = {node.get_id(): node for node in nodes}
for round_idx in range(n_rounds):
    for node in nodes:
        node.train()
    for node in nodes:
        node.aggregate_parameters(
            {
                peer: by_id[peer].get_parameters(round_id=round_idx)
                for peer in node.neighbors.recv_from
            }
        )
    meter.record_fidelity([node.algorithm_agent for node in nodes])
    meter.best_by_round.append(
        max(
            (b[1] for b in (n.algorithm_agent.best_so_far() for n in nodes) if b),
            default=float("-inf"),
        )
    )

report(
    f"serial, {node_config.neighbors.get('topology', 'fully_connected')}",
    topology,
    [node.algorithm_agent for node in nodes],
    meter,
    n_rounds,
    optimum=node_config.get("known_optimum", None),
)
