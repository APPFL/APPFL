"""
A relay for decentralized federated learning.

Run this once, on a host every participating site can reach. Sites that cannot be dialled --
behind NAT, or a firewall that permits only outbound connections -- publish their payloads
here, and the peers that collect from them fetch here. Sites that can serve do so directly and
never involve this process, so a federation can mix both kinds freely.

    python dfl/grpc/run_relay.py --config ./resources/configs/dfl/mnist/relay/relay.yaml
    python dfl/grpc/run_relay.py --config ... --server_uri localhost:50060   # same config, another port
"""

import argparse
import logging
from omegaconf import OmegaConf
from appfl.comm.grpc import GRPCRelayServicer, serve_relay

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--config",
    type=str,
    default="./resources/configs/dfl/mnist/relay/relay.yaml",
    help="Path to the configuration file.",
)
argparser.add_argument(
    "--server_uri",
    type=str,
    default=None,
    help="overrides the config's server_uri, so one config can be reused on another port.",
)
args = argparser.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

config = OmegaConf.load(args.config)
grpc_configs = config.get("comm_configs", {}).get("grpc_configs", {}) or {}
server_uri = args.server_uri or grpc_configs.get("server_uri", "localhost:50050")

serve_relay(
    GRPCRelayServicer(
        max_message_size=int(grpc_configs.get("max_message_size", 2 * 1024 * 1024)),
    ),
    server_uri=server_uri,
    use_ssl=bool(grpc_configs.get("use_ssl", False)),
    server_certificate=grpc_configs.get("server_certificate", None),
    server_certificate_key=grpc_configs.get("server_certificate_key", None),
    max_workers=int(grpc_configs.get("max_workers", 64)),
)
