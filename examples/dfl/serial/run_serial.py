"""
Serial simulation of Decentralized Federated Learning (DFL).

Every node lives in this one process, so there is no communication layer at all -- a node
reads its neighbor's published parameters directly. That makes this the control run: it is the
behavior the MPI and gRPC launchers have to reproduce, and it is the fastest way to see what a
topology does to convergence.

    # uses the default config, which is a 4-node fully connected graph of MNIST clients
    python dfl/serial/run_serial.py 
    
    # uses the default config, but overrides the number of nodes to 16
    python dfl/serial/run_serial.py --num_nodes 16 
    
    # uses a ring topology instead of the default fully connected one
    python dfl/serial/run_serial.py \
        --config ./resources/configs/dfl/mnist/simulation/node_0_ring.yaml

One config file describes a node completely, and this launcher reuses it for every node,
adjusting the identity and the data partition per node.
"""

import argparse
import copy
import warnings
from omegaconf import OmegaConf
from appfl.agent import DFLNodeAgent
from appfl.decentralized.neighbor import (
    build_topology_from_config,
    resolve_all_neighbors_from_topology,
)

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--config", type=str, default="./resources/configs/dfl/mnist/simulation/node_0_full.yaml"
)
argparser.add_argument("--num_nodes", type=int, default=None)
argparser.add_argument("--num_epochs", type=int, default=None)
args = argparser.parse_args()

node_config = OmegaConf.load(args.config)
if args.num_nodes is not None:
    node_config.neighbors.num_nodes = args.num_nodes
if args.num_epochs is not None:
    node_config.num_epochs = args.num_epochs

# Build the graph once, and immediately reduce it to one neighbor list per node. The graph is
# a convenience of simulation -- this launcher owns every node, so it can afford to know the
# whole federation. 
topology = build_topology_from_config(node_config.neighbors)
node_ids = topology.node_ids
num_nodes = len(node_ids)
if not topology.is_connected():
    warnings.warn(
        f"topology '{node_config.neighbors.topology}' is not connected: {topology.describe()}. "
        + (
            "Some nodes influence others without ever being influenced back, so the run will "
            "not reach one consensus."
            if topology.directed
            else "Each connected component reaches its own consensus, so this is several "
            "independent federations rather than one."
        ),
        UserWarning,
        stacklevel=2,
    )
neighbors = resolve_all_neighbors_from_topology(topology, node_ids)
print(f"topology: {node_config.neighbors.topology} {topology.describe()}")
for node_id in node_ids:
    view = neighbors[node_id]
    line = f"  {node_id} receives from {view.recv_from_ids}"
    if topology.directed:
        line += f", serves {view.send_to}"
    print(line)

# One agent per node, from one config: set the identity and the data slice, leave the rest.
node_agents = []
for index, node_id in enumerate(node_ids):
    config = copy.deepcopy(node_config)
    config.node_id = node_id
    config.train_configs.logging_id = node_id
    config.data_configs.dataset_kwargs.num_clients = num_nodes
    config.data_configs.dataset_kwargs.client_id = index
    config.data_configs.dataset_kwargs.visualization = index == 0
    node_agents.append(
        DFLNodeAgent(dfl_node_agent_config=config, neighbors=neighbors[node_id])
    )
agents_by_id = {agent.get_id(): agent for agent in node_agents}

# The DFL round: every node trains, then every node mixes with its neighbors.
for epoch in range(int(node_config.num_epochs)):
    for agent in node_agents:
        agent.train()
    neighbor_models = {
        agent.get_id(): {
            neighbor_id: agents_by_id[neighbor_id].get_parameters(
                round_id=epoch, requester_id=agent.get_id()
            )
            for neighbor_id in agent.neighbors.recv_from_ids
        }
        for agent in node_agents
    }
    for agent in node_agents:
        agent.aggregate_parameters(neighbor_models[agent.get_id()])
