# Decentralized federated learning

There is no server. Every participant is a **node** that trains on its own data, serves its local model to the neighbors it allows, and averages what it receives back into its own model. These examples train a CNN on MNIST split four ways, over three transports: serial for a control run, MPI for scale, gRPC for real deployment across machines.

```text
examples/dfl/
├── serial/run_serial.py   every node in one process -- the control run
├── mpi/run_mpi.py         one node per rank
└── grpc/
    ├── run_node.py        one process per site, over the network
    └── run_relay.py       for sites that cannot be dialled; run once, on a reachable host
```

All commands run from `examples/`.

## Simulation: one config, reused for every node

```bash
python dfl/serial/run_serial.py                                                         # 4 nodes, fully connected
python dfl/serial/run_serial.py --config ./resources/configs/dfl/mnist/simulation/node_0_ring.yaml
python dfl/serial/run_serial.py --num_nodes 16

mpirun -n 4 python dfl/mpi/run_mpi.py                                                   # -n decides the size
mpirun -n 8 python dfl/mpi/run_mpi.py --config ./resources/configs/dfl/mnist/simulation/node_0_ring.yaml
```

Under MPI the rank count decides the federation size, and a config asking for a different `num_nodes` is an error rather than an override.

## Deployment: one config per site

Four terminals, or four institutions. Nodes may start in any order and minutes apart — each serves before it dials, and a peer that is not up yet is waited for rather than failing.

```bash
python dfl/grpc/run_node.py --config ./resources/configs/dfl/mnist/distributed/node_0.yaml
python dfl/grpc/run_node.py --config ./resources/configs/dfl/mnist/distributed/node_1.yaml
python dfl/grpc/run_node.py --config ./resources/configs/dfl/mnist/distributed/node_2.yaml
python dfl/grpc/run_node.py --config ./resources/configs/dfl/mnist/distributed/node_3.yaml
```

The shipped set is four sites on localhost, so this runs on one machine; replacing the URIs with real hostnames changes nothing else.

A site behind NAT cannot be dialled, so it publishes to a relay instead and its peers collect from there. Start the relay first:

```bash
python dfl/grpc/run_relay.py --config ./resources/configs/dfl/mnist/relay/relay.yaml
python dfl/grpc/run_node.py  --config ./resources/configs/dfl/mnist/relay/node_0.yaml   # serves itself
python dfl/grpc/run_node.py  --config ./resources/configs/dfl/mnist/relay/node_2.yaml   # publishes to the relay
```

A node either serves its own payloads (`server_uri`) or publishes them (`relay_server_uri`), never both. Sites 0 and 1 serve directly and never involve the relay; sites 2 and 3 publish through it.

## Options

Every launcher takes `--config`, `--num_epochs`, and `--save_parameters <dir>`, which writes each node's final parameters as `<node_id>.pt`. `--num_nodes` is serial-only. `run_relay.py` takes `--config` and `--server_uri`.

## Configs

```text
resources/configs/dfl/mnist/
├── simulation/          one config, reused for every node by the serial and MPI launchers
│   ├── node_0_full.yaml   neighbors derived from a fully connected graph
│   └── node_0_ring.yaml   the same, on a ring
├── distributed/         one config per site, read by dfl/grpc/run_node.py
│   └── node_{0..3}.yaml   neighbors named explicitly, with endpoints
└── relay/               the same four sites, two of them publishing through a relay
    ├── node_{0..3}.yaml
    └── relay.yaml         read by dfl/grpc/run_relay.py
```

**One file per node, and that file is complete** — there is no server, so a node's config describes everything it needs: who it talks to, what it trains, on what data. Apart from that, it is an ordinary APPFL client config, so trainer, model, dataset, compression and logging work exactly as in `examples/serial`.

The only part that is new is the `neighbors` block, which has two modes. `explicit` is what a deployment uses:

```yaml
neighbors:
  mode: "explicit"
  send_to: ["Node1", "Node2"]        # who may request MY model; anyone else is refused
  recv_from:                         # whose models I fetch and average
    - node_id: "Node1"
      server_uri: "site-b.example.org:50051"   # their address, or the relay they publish through
    - node_id: "Node2"
      server_uri: "site-c.example.org:50051"
```

The two lists must name the same peers, since averaging reaches consensus only on an undirected graph; a config that disagrees with itself is rejected unless `allow_directed: true` says the asymmetry is deliberate. `from_topo` is for simulation, where writing out lists for a thousand nodes is untenable:

```yaml
neighbors:
  mode: "from_topo"
  topology: "fully_connected"   # fully_connected | ring | star | random_geometric | custom
  topology_kwargs: {}
  num_nodes: 4
```

The serial and MPI launchers require `from_topo` and the gRPC launcher requires `explicit`, because a deployed site names its neighbors and never holds a roster. Both modes produce the same neighbor list and the agent cannot tell which was used, which is what keeps the serial run a valid control for the distributed one.

## Notes

On a fully connected graph every node averages all `n` models with weight `1/n`, so they all end each round identical — and equal to what centralized FedAvg would produce. Sparser graphs trade that off deliberately: a ring mixes with two neighbors regardless of federation size, so per-node cost stops growing with `n` but information needs O(n) rounds to cross.

The three transports produce bitwise identical parameters on the same config, which is what `--save_parameters` is for. **This requires `train_data_shuffle: false`, which none of the shipped configs set** — with shuffling on, serial and MPI legitimately differ by around 1e-1 after three rounds, because serial's nodes share one process RNG while each MPI rank is seeded identically and shuffles the same way.

To see a topology rather than read its edge lists:

```python
from appfl.decentralized.topology import build_topology

topology = build_topology("ring", [f"Node{i}" for i in range(8)], k=2)
topology.draw("ring.png")
print(topology.to_ascii())   # adjacency matrix, row sends to column; summary only above 24 nodes
```

For the design behind all of this — round-tagged publication, access control, mixing weights, directed graphs — see [`appfl.decentralized`](../../src/appfl/decentralized) and [`DFLNodeAgent`](../../src/appfl/agent/dfl_node.py).
