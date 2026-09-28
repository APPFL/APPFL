"""The token relay -- run this once, on a host every participating site can reach.

Sites do not connect to each other. Each dials out to this process, which applies the
topology and hands every site exactly its neighbors' tokens. That is a deliberate choice, not
a shortcut:

  * DOE sites generally cannot accept inbound connections. A full peer-to-peer mesh would
    need N x N firewall exceptions and N server certificates. Every site dialing out to one
    endpoint is the shape that actually gets deployed, and the shape AmSC federated identity
    is built around.
  * The *algorithm* stays decentralized regardless. There is no global model, no pooled data,
    and no agent sees anything beyond its own neighbors' tokens. This process routes bytes it
    never interprets -- a switchboard, not an aggregator.

Note what this launcher does NOT construct: a ServerAgent. There is no model to hold, no
aggregator, no scheduler. That absence is the clearest statement of what a decentralized run
needs from a server, which is almost nothing. It reads only the federation config, because
the graph is the one thing the relay has to know and the only thing it is entitled to.

    python grpc/run_relay.py --server_uri localhost:50051

Add --use_ssl with certificates for anything crossing a real network.
"""

import argparse
import sys
from pathlib import Path

#: ``examples/decentralized``. Relative resource paths in the configs resolve against it, so
#: a launcher can be invoked from here or from the repository root.
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_ROOT / "resources"))

from appfl.comm.grpc import serve

from appfl.decentralized import RelayServer, create_topology, load_federation_config
from appfl.decentralized.exchange.grpc_servicer import RelayServicer



def report_shutdown(action_count: int) -> None:
    """Printed by the servicer on Ctrl-C. Lives here, not in the library, because a library
    should not write to stdout."""
    print(f"\nrelay handled {action_count} actions")


def main() -> None:
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Only the federation config: the relay is entitled to the graph and nothing else.
    argparser.add_argument(
        "--federation_config", type=str, default="./resources/configs/toy1d/federation_adko.yaml"
    )
    argparser.add_argument("--server_uri", type=str, default="localhost:50051")
    argparser.add_argument("--use_ssl", action="store_true")
    argparser.add_argument("--server_certificate", type=str, default=None)
    argparser.add_argument("--server_certificate_key", type=str, default=None)
    args = argparser.parse_args()

    federation_config = load_federation_config(args.federation_config, EXAMPLE_ROOT)
    topology = create_topology(federation_config)
    relay = RelayServer(topology)
    print(f"relay up at {args.server_uri}")
    print(f"topology: {topology.describe()}")
    for agent_id in topology.agent_ids:
        print(f"  {agent_id} -> neighbors {topology.neighbors(agent_id)}")

    serve(
        RelayServicer(relay, on_shutdown=report_shutdown),
        server_uri=args.server_uri,
        use_ssl=args.use_ssl,
        server_certificate=args.server_certificate,
        server_certificate_key=args.server_certificate_key,
    )


if __name__ == "__main__":
    main()
