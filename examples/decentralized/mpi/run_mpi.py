"""One node of a decentralized optimization federation per MPI rank.

# run from `examples` folder: `-n` decides the federation size
mpirun -n 4 python decentralized/mpi/run_mpi.py
mpirun -n 8 python decentralized/mpi/run_mpi.py --num_rounds 10

# ADKO v2 -- the config names the algorithm
mpirun -n 4 python decentralized/mpi/run_mpi.py \
    --config ./decentralized/resources/configs/suzuki_v2/simulation/node_0_het.yaml
"""


import sys
import copy
import time
import argparse
from mpi4py import MPI
from omegaconf import OmegaConf
from appfl.agent import DeOptNodeAgent
from appfl.comm.mpi import MPIPeerCommunicator
from appfl.decentralized.budget import CommBudget
from appfl.decentralized.neighbor import (
    build_topology_from_config,
    resolve_all_neighbors_from_topology,
)

sys.path.insert(0, "./decentralized/resources")

from algorithms import load_algorithm  # noqa: E402
from llm_cli import add_llm_arguments  # noqa: E402
from reporting import (  # noqa: E402
    describe_slice,
    format_point,
    health,
    print_config,
    print_llm,
    print_totals,
)
from run_record import run_record, write_run_record  # noqa: E402

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--config", type=str, default="./decentralized/resources/configs/toy1d/simulation/node_0.yaml"
)
argparser.add_argument("--num_rounds", type=int, default=None)
argparser.add_argument("--seed", type=int, default=None, help="override the config's run seed")
argparser.add_argument(
    "--results_path",
    type=str,
    default=None,
    help="write an ADKO v2 run record here, from rank 0 (the reference's JSON schema)",
)
add_llm_arguments(argparser)
args = argparser.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

node_config = OmegaConf.load(args.config)
if args.num_rounds is not None:
    node_config.num_epochs = args.num_rounds
if args.seed is not None:
    node_config.seed = args.seed
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
algorithm = load_algorithm(node_config)
meter = algorithm.meter
budget_configs = node_config.get("comm_configs", {}).get("budget_configs", {}) or {}
budget = CommBudget(
    bits_per_neighbor_per_round=budget_configs.get("bits_per_neighbor_per_round", None),
    tokens_per_neighbor_per_round=int(
        budget_configs.get("tokens_per_neighbor_per_round", 1)
    ),
)

algorithm_agent = algorithm.create_agent(config, neighbors, args, topology=topology)
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

start = time.time()
for round_idx in range(n_rounds):
    agent.train()
    tokens = agent.get_parameters(round_id=round_idx)
    peer_tokens = communicator.exchange(round_idx, tokens)
    agent.aggregate_parameters(peer_tokens)
    algorithm.end_round([algorithm_agent])
wall_seconds = time.time() - start
algorithm.close([algorithm_agent])

# Reduce onto rank 0 for reporting purposes.
gathered = comm.gather(
    {
        "agent_id": algorithm_agent.agent_id,
        "slice": describe_slice(algorithm_agent),
        "best": algorithm_agent.best_so_far(),
        "health": health(algorithm_agent),
        "meter": meter,
        "llm": (
            algorithm_agent.language_model.stats()
            if getattr(algorithm_agent, "language_model", None) is not None
            else None
        ),
        "trace": (
            algorithm_agent.trace() if hasattr(algorithm_agent, "trace") else None
        ),
    },
    root=0,
)
if rank != 0:
    raise SystemExit(0)

total_meter = type(meter)()
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
            f"{peer_info['health']}"
        )
print_totals(total_meter, n_rounds, optimum=node_config.get("known_optimum", None))
print_llm([peer_info["llm"] for peer_info in gathered])
if args.results_path:
    if any(peer_info["trace"] is None for peer_info in gathered):
        raise SystemExit("--results_path: only adko_v2 agents write run records")
    write_run_record(
        args.results_path,
        run_record(
            [peer_info["trace"] for peer_info in gathered],
            seed=int(node_config.get("seed", 0)),
            setting=getattr(algorithm_agent.space, "setting", None),
            n_rounds=n_rounds,
            config=OmegaConf.to_container(node_config, resolve=True),
            wall_seconds=wall_seconds,
        ),
    )
