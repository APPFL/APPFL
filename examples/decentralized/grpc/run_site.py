"""One site of a decentralized optimization federation, over gRPC.

Run this once per site, each with its own config. A site serves its tokens to the peers that
collect from them and dials out to the peers it collects from.

# run from `examples` -- four terminals, or four institutions
python decentralized/grpc/run_site.py --config ./decentralized/resources/configs/toy1d/distributed/node_0.yaml
python decentralized/grpc/run_site.py --config ./decentralized/resources/configs/toy1d/distributed/node_1.yaml
python decentralized/grpc/run_site.py --config ./decentralized/resources/configs/toy1d/distributed/node_2.yaml
python decentralized/grpc/run_site.py --config ./decentralized/resources/configs/toy1d/distributed/node_3.yaml

A site that cannot accept inbound connections publishes to a relay instead, and its peers
collect from there. Both kinds mix freely in one federation, and this launcher cannot tell
which it is running -- the difference is entirely in the config:

python decentralized/grpc/run_relay.py
python decentralized/grpc/run_site.py --config ./decentralized/resources/configs/toy1d/relay/node_0.yaml
python decentralized/grpc/run_site.py --config ./decentralized/resources/configs/toy1d/relay/node_1.yaml
python decentralized/grpc/run_site.py --config ./decentralized/resources/configs/toy1d/relay/node_2.yaml
python decentralized/grpc/run_site.py --config ./decentralized/resources/configs/toy1d/relay/node_3.yaml

Sites may start in any order: each serves before it dials, and a connection to a peer that is
not up yet waits rather than failing. The config must use `neighbors.mode: explicit` -- a
deployed site names its neighbors and never holds a roster, which is exactly what
distinguishes this from the simulation launchers.

The config also names the algorithm; ADKO v2's HET arm deploys the same way:

python decentralized/grpc/run_site.py --config ./decentralized/resources/configs/suzuki_v2/distributed/node_0.yaml
"""

import sys
import time
import argparse
from omegaconf import OmegaConf
from appfl.agent import DeOptNodeAgent
from appfl.comm.grpc import GRPCPeerCommunicator
from appfl.decentralized.budget import CommBudget
from appfl.decentralized.neighbor import resolve_neighbors

sys.path.insert(0, "./decentralized/resources")

from algorithms import load_algorithm  # noqa: E402
from llm_cli import add_llm_arguments  # noqa: E402
from reporting import describe_slice, format_point, health, print_totals  # noqa: E402
from run_record import run_record, write_run_record  # noqa: E402

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--config",
    type=str,
    default="./decentralized/resources/configs/toy1d/distributed/node_0.yaml",
)
argparser.add_argument("--num_rounds", type=int, default=None)
argparser.add_argument("--seed", type=int, default=None, help="override the config's run seed")
argparser.add_argument(
    "--results_path",
    type=str,
    default=None,
    help="write this site's ADKO v2 run record here (the reference's JSON schema)",
)
add_llm_arguments(argparser)
args = argparser.parse_args()

node_config = OmegaConf.load(args.config)
if args.num_rounds is not None:
    node_config.num_epochs = args.num_rounds
if args.seed is not None:
    node_config.seed = args.seed
n_rounds = int(node_config.num_epochs)

node_id = str(node_config.node_id)
neighbors = resolve_neighbors(node_config.neighbors, node_id)

algorithm = load_algorithm(node_config)
meter = algorithm.meter
budget_configs = node_config.get("comm_configs", {}).get("budget_configs", {}) or {}
budget = CommBudget(
    bits_per_neighbor_per_round=budget_configs.get("bits_per_neighbor_per_round", None),
    tokens_per_neighbor_per_round=int(
        budget_configs.get("tokens_per_neighbor_per_round", 1)
    ),
)

# No topology: this site has none, and does not need one. 
algorithm_agent = algorithm.create_agent(node_config, neighbors, args)
agent = DeOptNodeAgent(
    algorithm_agent=algorithm_agent,
    dfl_node_agent_config=node_config,
    neighbors=neighbors,
    budget=budget,
    meter=meter,
)

grpc_configs = node_config.get("comm_configs", {}).get("grpc_configs", {}) or {}
communicator = GRPCPeerCommunicator(
    node_id=node_id,
    # Exactly one of the two, and which one is this site's own business. Its peers' `recv_from`
    # already says where to come looking, so neither they nor this launcher need to know.
    server_uri=grpc_configs.get("server_uri", None),
    relay_server_uri=grpc_configs.get("relay_server_uri", None),
    send_to=neighbors.send_to,
    recv_from=neighbors.recv_from,
    use_ssl=bool(grpc_configs.get("use_ssl", False)),
    server_certificate=grpc_configs.get("server_certificate", None),
    server_certificate_key=grpc_configs.get("server_certificate_key", None),
    root_certificate=grpc_configs.get("root_certificate", None),
    max_message_size=int(grpc_configs.get("max_message_size", 2 * 1024 * 1024)),
    logger=agent.logger,
)

start = time.time()
with communicator:
    for round_idx in range(n_rounds):
        agent.train()
        tokens = agent.get_parameters(round_id=round_idx)
        peer_tokens = communicator.exchange(round_idx, tokens)
        agent.aggregate_parameters(peer_tokens)
        algorithm.end_round([algorithm_agent])
wall_seconds = time.time() - start
algorithm.close([algorithm_agent])

best = algorithm_agent.best_so_far()
print(f"\n=== grpc, {node_id} ===")
if best is not None:
    print(
        f"  {node_id} {describe_slice(algorithm_agent)}  "
        f"best x={format_point(best[0])} yield={best[1]:.1f}  "
        f"{health(algorithm_agent)}"
    )
print_totals(meter, n_rounds, optimum=node_config.get("known_optimum", None))
if args.results_path:
    if not hasattr(algorithm_agent, "trace"):
        raise SystemExit("--results_path: only adko_v2 agents write run records")
    write_run_record(
        args.results_path,
        run_record(
            [algorithm_agent.trace()],
            seed=int(node_config.get("seed", 0)),
            setting=getattr(algorithm_agent.space, "setting", None),
            n_rounds=n_rounds,
            config=OmegaConf.to_container(node_config, resolve=True),
            wall_seconds=wall_seconds,
        ),
    )
