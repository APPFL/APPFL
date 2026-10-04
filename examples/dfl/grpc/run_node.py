"""
One node of a decentralized federation, over gRPC.

Run this once per site, each with its own config. Every node serves its own model to the peers
that collect from it and dials out to the peers it collects from; there is no coordinator and
no process that sees the whole federation.

    # four terminals, or four institutions
    python dfl/grpc/run_node.py --config ./resources/configs/dfl/mnist/distributed/node_0.yaml
    python dfl/grpc/run_node.py --config ./resources/configs/dfl/mnist/distributed/node_1.yaml
    python dfl/grpc/run_node.py --config ./resources/configs/dfl/mnist/distributed/node_2.yaml
    python dfl/grpc/run_node.py --config ./resources/configs/dfl/mnist/distributed/node_3.yaml

Nodes may start in any order: each serves before it dials, and a connection to a peer that is
not up yet waits rather than failing. The config must use `neighbors.mode: explicit` -- a
deployed site names its neighbors and never holds a roster, which is exactly what distinguishes
this from the simulation launchers.
"""

import argparse
from pathlib import Path
from omegaconf import OmegaConf
from appfl.agent import DFLNodeAgent
from appfl.comm.grpc import GRPCPeerCommunicator
from appfl.decentralized.neighbor import resolve_neighbors

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--config",
    type=str,
    default="./resources/configs/dfl/mnist/distributed/node_0.yaml",
)
argparser.add_argument("--num_epochs", type=int, default=None)
argparser.add_argument(
    "--save_parameters",
    type=str,
    default=None,
    help="directory to write this node's final parameters to, as <node_id>.pt",
)
args = argparser.parse_args()

config = OmegaConf.load(args.config)
if args.num_epochs is not None:
    config.num_epochs = args.num_epochs

node_id = str(config.node_id)
neighbors = resolve_neighbors(config.neighbors, node_id)
agent = DFLNodeAgent(dfl_node_agent_config=config, neighbors=neighbors)

grpc_configs = config.get("comm_configs", {}).get("grpc_configs", {}) or {}

communicator = GRPCPeerCommunicator(
    node_id=node_id,
    server_uri=grpc_configs.get("server_uri", "localhost:50051"),
    send_to=neighbors.send_to,
    recv_from=neighbors.recv_from,
    use_ssl=bool(grpc_configs.get("use_ssl", False)),
    server_certificate=grpc_configs.get("server_certificate", None),
    server_certificate_key=grpc_configs.get("server_certificate_key", None),
    root_certificate=grpc_configs.get("root_certificate", None),
    max_message_size=int(grpc_configs.get("max_message_size", 2 * 1024 * 1024)),
    logger=agent.logger,
)

with communicator:
    for epoch in range(int(config.num_epochs)):
        agent.train()
        neighbor_models = communicator.exchange(
            epoch, agent.get_parameters(round_id=epoch)
        )
        agent.aggregate_parameters(neighbor_models)

if args.save_parameters is not None:
    import torch

    output = Path(args.save_parameters)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(agent.get_parameters(), output / f"{node_id}.pt")
