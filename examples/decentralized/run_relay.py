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

Note what this servicer does NOT construct: a ServerAgent. There is no model to hold, no
aggregator, no scheduler. That absence is the clearest statement of what a decentralized run
needs from a server, which is almost nothing.

    python examples/decentralized/run_relay.py --server-uri localhost:50051

Add --use-ssl with certificates for anything crossing a real network.
"""

import argparse

from appfl.comm.grpc import serve

from appfl.decentralized import RelayServer
from appfl.decentralized.exchange.grpc_servicer import RelayServicer

from benchmark import make_topology


def report_shutdown(action_count: int) -> None:
    """Printed by the servicer on Ctrl-C. Lives here, not in the library, because a library
    should not write to stdout."""
    print(f"\nrelay handled {action_count} actions")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-uri", default="localhost:50051")
    parser.add_argument("--topology", default="fully_connected")
    parser.add_argument("--use-ssl", action="store_true")
    parser.add_argument("--server-certificate", default=None)
    parser.add_argument("--server-certificate-key", default=None)
    args = parser.parse_args()

    topology = make_topology(args.topology)
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
