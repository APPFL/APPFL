"""One node of a decentralized optimization federation per MPI rank.

# run from `examples` folder: `-n` decides the federation size
mpirun -n 4 python decentralized/mpi/run_mpi.py
mpirun -n 8 python decentralized/mpi/run_mpi.py --num_rounds 10
"""


import sys
import copy
import argparse
from mpi4py import MPI
from omegaconf import OmegaConf
from appfl.agent import DeOptNodeAgent
from appfl.comm.mpi import MPIPeerCommunicator
from appfl.decentralized.algorithm.adko import create_agent
from appfl.decentralized.algorithm.adko.metrics import ADKOMeter
from appfl.decentralized.budget import CommBudget
from appfl.decentralized.neighbor import (
    build_topology_from_config,
    resolve_all_neighbors_from_topology,
)

sys.path.insert(0, "./decentralized/resources")

from llm_cli import add_llm_arguments, llm_config_from_args  # noqa: E402
from reporting import (  # noqa: E402
    describe_slice,
    format_point,
    print_config,
    print_llm,
    print_totals,
)

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--config", type=str, default="./decentralized/resources/configs/toy1d/simulation/node_0.yaml"
)
argparser.add_argument("--num_rounds", type=int, default=None)
add_llm_arguments(argparser)
args = argparser.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

node_config = OmegaConf.load(args.config)
if args.num_rounds is not None:
    node_config.num_epochs = args.num_rounds
# The rank count decides the federation size
node_config.neighbors.num_nodes = comm.Get_size()
node_config.space_configs.space_kwargs.num_agents = comm.Get_size()
n_rounds = int(node_config.num_epochs)

topology = build_topology_from_config(node_config.neighbors)
node_ids = topology.node_ids
node_id = node_ids[rank]
neighbors = resolve_all_neighbors_from_topology(topology, node_ids)[node_id]

# This rank's node: the same config as every other, with only the identity and the slice of the design space differing.
config = copy.deepcopy(node_config)
config.node_id = node_id
config.space_configs.space_kwargs.agent_index = rank
config.train_configs.logging_id = node_id

# One meter per rank, merged onto rank 0 at the end -- a rank can only count its own traffic.
meter = ADKOMeter()
budget_configs = node_config.get("comm_configs", {}).get("budget_configs", {}) or {}
budget = CommBudget(
    bits_per_neighbor_per_round=budget_configs.get("bits_per_neighbor_per_round", None),
    tokens_per_neighbor_per_round=int(
        budget_configs.get("tokens_per_neighbor_per_round", 1)
    ),
)

algorithm_agent = create_agent(
    config, neighbors, meter, llm_config=llm_config_from_args(args), topology=topology
)
agent = DeOptNodeAgent(
    algorithm_agent=algorithm_agent,
    dfl_node_agent_config=config,
    neighbors=neighbors,
    budget=budget,
    meter=meter,
)

communicator = MPIPeerCommunicator(
    comm,
    node_id=node_id,
    node_ids=node_ids,
    send_to=neighbors.send_to,
    recv_from=list(neighbors.recv_from), # recv_from is a dict_keys view, but MPIPeerCommunicator wants a list of ids
)

for round_idx in range(n_rounds):
    agent.train()
    tokens = agent.get_parameters(round_id=round_idx)
    peer_tokens = communicator.exchange(round_idx, tokens)
    agent.aggregate_parameters(peer_tokens)
    meter.record_fidelity([algorithm_agent])

# Reduce onto rank 0 for reporting purposes.
gathered = comm.gather(
    {
        "agent_id": algorithm_agent.agent_id,
        "slice": describe_slice(algorithm_agent),
        "best": algorithm_agent.best_so_far(),
        "eta_bar": algorithm_agent.mean_token_fidelity(),
        "meter": meter,
        "llm": (
            algorithm_agent.language_model.stats()
            if algorithm_agent.language_model is not None
            else None
        ),
    },
    root=0,
)
if rank != 0:
    raise SystemExit(0)

total_meter = ADKOMeter()
for peer_info in gathered:
    total_meter.merge(peer_info["meter"])
    if peer_info["best"] is not None:
        total_meter.best_by_round.append(peer_info["best"][1])
print(f"\n=== mpi, {node_config.neighbors.get('topology', 'fully_connected')} ===")
print(f"topology            : {topology.describe()}")
print_config([algorithm_agent])
for peer_rank, peer_info in enumerate(gathered):
    if peer_info["best"] is not None:
        print(
            f"  {peer_info['agent_id']} (rank {peer_rank}) {peer_info['slice']}  "
            f"best x={format_point(peer_info['best'][0])} yield={peer_info['best'][1]:.1f}  "
            f"eta_bar={peer_info['eta_bar']:.3f}"
        )
print_totals(total_meter, n_rounds, optimum=node_config.get("known_optimum", None))
print_llm([peer_info["llm"] for peer_info in gathered])
