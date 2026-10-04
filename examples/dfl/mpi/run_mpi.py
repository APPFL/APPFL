"""
MPI simulation of Decentralized Federated Learning (DFL).

One node per rank, exchanging models directly with its graph neighbors.

    # uses the default config, which is a 4-node fully connected graph of MNIST clients
    mpirun -n 4 python dfl/mpi/run_mpi.py

    # uses a ring topology instead of the default fully connected one with 8 nodes
    mpirun -n 8 python dfl/mpi/run_mpi.py \
        --config ./resources/configs/dfl/mnist/simulation/node_0_ring.yaml

Every rank reads the same config and derives its own identity from its rank, so `-n` must
equal the node count the config asks for -- the communicator refuses a mismatch.
"""

import argparse
import copy
import warnings
from pathlib import Path

from mpi4py import MPI
from omegaconf import OmegaConf

from appfl.agent import DFLNodeAgent
from appfl.comm.mpi import MPIPeerCommunicator
from appfl.decentralized.neighbor import (
    build_topology_from_config,
    resolve_all_neighbors_from_topology,
)

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--config",
    type=str,
    default="./resources/configs/dfl/mnist/simulation/node_0_full.yaml",
)
argparser.add_argument("--num_epochs", type=int, default=None)
argparser.add_argument(
    "--save_parameters",
    type=str,
    default=None,
    help="directory to write each node's final parameters to, as <node_id>.pt",
)
args = argparser.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

node_config = OmegaConf.load(args.config)
if args.num_epochs is not None:
    node_config.num_epochs = args.num_epochs
# The rank count decides the federation size; a config that disagrees is a mistake, not an
# override. Setting it here means the topology is built over exactly the ranks that exist.
node_config.neighbors.num_nodes = comm.Get_size()

topology = build_topology_from_config(node_config.neighbors)
node_ids = topology.node_ids
node_id = node_ids[rank]
neighbors = resolve_all_neighbors_from_topology(topology, node_ids)[node_id]

if rank == 0:
    if not topology.is_connected():
        warnings.warn(
            f"topology '{node_config.neighbors.topology}' is not connected: "
            f"{topology.describe()}. Each component reaches its own consensus, so this is "
            f"several independent federations rather than one.",
            UserWarning,
            stacklevel=2,
        )
    print(f"topology: {node_config.neighbors.topology} {topology.describe()}")

# This rank's node: the same config as every other, with only the identity and the data slice differing
config = copy.deepcopy(node_config)
config.node_id = node_id
config.train_configs.logging_id = node_id
config.data_configs.dataset_kwargs.num_clients = len(node_ids)
config.data_configs.dataset_kwargs.client_id = rank
config.data_configs.dataset_kwargs.visualization = False
agent = DFLNodeAgent(dfl_node_agent_config=config, neighbors=neighbors)

communicator = MPIPeerCommunicator(
    comm,
    node_id=node_id,
    node_ids=node_ids,
    send_to=neighbors.send_to,
    recv_from=list(neighbors.recv_from),
)

for epoch in range(int(node_config.num_epochs)):
    agent.train()
    neighbor_models = communicator.exchange(epoch, agent.get_parameters(round_id=epoch))
    agent.aggregate_parameters(neighbor_models)

if args.save_parameters is not None:
    import torch

    output = Path(args.save_parameters)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(agent.get_parameters(), output / f"{node_id}.pt")
