# Decentralized optimization on APPFL

A federation of agents that each optimize over their own slice of a design space and exchange **knowledge tokens** — compact statements about what worked — rather than models or raw data. [ADKO](../../src/appfl/decentralized/algorithm/adko) is the reference algorithm; the infrastructure under it is APPFL's decentralized FL layer, unchanged.

A node here is an [`appfl.agent.DeOptNodeAgent`](../../src/appfl/agent/deopt_node.py): a `DFLNodeAgent` whose payload is tokens instead of model parameters. That is the whole adaptation, because a DFL round and an optimization round are the same three steps:

| APPFL | ADKO |
| --- | --- |
| `train()` | `act(round)` — propose, evaluate, emit a token |
| `get_parameters(r)` | the token itself |
| `aggregate_parameters(peers)` | `observe(tokens)` |

```text
examples/decentralized/
├── serial/run_serial.py    every node in one process -- the control run
├── mpi/run_mpi.py          one node per rank, peer-to-peer -- the path that scales
└── grpc/
    ├── run_site.py         one process per site, over the network -- the deployment path
    └── run_relay.py        for sites that cannot accept inbound connections
```

All commands are run from `examples`. **Which launcher you run decides the transport; which config you pass decides the experiment.** Two experiments are provided:

| | `toy1d` | `suzuki` |
| --- | --- | --- |
| what it is | a synthetic 1-D yield curve | the Suzuki–Miyaura coupling study from the ADKO paper |
| a design point | a float in `[0, 1]` | five category ids — 3,696 measured reactions |
| the oracle | a closed-form function | the measured `suzuki_edbo` lookup table |
| extra dependencies | none | BoTorch, GPyTorch, Olympus |
| what it is for | checking that the transports agree | reproducing a published result |

`toy1d` is the default everywhere, because it runs in seconds and is what the cross-transport equivalence below is measured on. The next three sections apply unchanged to `suzuki` — pass its config instead.


A third experiment, `cifar10`, reproduces the ADKO paper's CIFAR-10 architecture search:

| | `cifar10` |
| --- | --- |
| what it is | the CIFAR-10 architecture search study from the ADKO paper |
| a design point | seven floats in `[0, 1]` — a ResNet and its training recipe |
| the oracle | training that ResNet on the agent's own CIFAR-10 shard |
| extra dependencies | none beyond APPFL's `examples` extra; a GPU in practice |
| what it is for | reproducing a published result |

## Simulation: one file describes every node

`neighbors.mode: from_topo`. A launcher that owns the whole federation derives the graph and assigns matching node ids, so nodes 1..n-1 come from node 0's config.

```bash
python decentralized/serial/run_serial.py
python decentralized/serial/run_serial.py --num_nodes 8 --num_rounds 10

mpirun -n 4 python decentralized/mpi/run_mpi.py
mpirun -n 8 python decentralized/mpi/run_mpi.py --num_rounds 10
```

`-n` decides the federation size: the topology is built over exactly the ranks that exist. Each node sends one message per out-neighbor and receives one per in-neighbor — nothing else — so a ring costs two messages per node whether the federation has 8 nodes or 8000.

## Deployment: one file per site, and that file is complete

`neighbors.mode: explicit`. A deployed site names its own neighbors and where to reach each one, and holds nothing else about the federation — no roster, no graph. Four terminals, or four institutions:

```bash
python decentralized/grpc/run_site.py --config ./decentralized/resources/configs/toy1d/distributed/node_0.yaml
python decentralized/grpc/run_site.py --config ./decentralized/resources/configs/toy1d/distributed/node_1.yaml
python decentralized/grpc/run_site.py --config ./decentralized/resources/configs/toy1d/distributed/node_2.yaml
python decentralized/grpc/run_site.py --config ./decentralized/resources/configs/toy1d/distributed/node_3.yaml
```

Sites may start in any order and minutes apart: each serves before it dials, and a connection to a peer that is not up yet waits rather than failing.

## Mixed: some sites served, some relayed

A site behind NAT, or a firewall that permits only outbound connections, cannot be dialled. It publishes to a **relay** instead, and its peers collect from there. Sites that can serve do so directly and never involve the relay, so one federation mixes both kinds:

```bash
python decentralized/grpc/run_relay.py                       # on a host every site can reach
python decentralized/grpc/run_site.py --config ./decentralized/resources/configs/toy1d/relay/node_0.yaml
python decentralized/grpc/run_site.py --config ./decentralized/resources/configs/toy1d/relay/node_1.yaml
python decentralized/grpc/run_site.py --config ./decentralized/resources/configs/toy1d/relay/node_2.yaml
python decentralized/grpc/run_site.py --config ./decentralized/resources/configs/toy1d/relay/node_3.yaml
```

In `resources/configs/toy1d/relay/`, sites 0 and 1 serve and sites 2 and 3 are relayed. **The launcher is the same, and cannot tell which it is running.** Two things in a site's config carry the whole difference:

- it sets `grpc_configs.server_uri` if it serves, or `grpc_configs.relay_server_uri` if it publishes — never both;
- its peers' `recv_from` entries point at the relay for relayed peers and at the peer itself otherwise. The collecting side sends the same request either way.

`send_to` is an access list, not just routing. A publisher declares it on **every** publish, so the relay enforces exactly the edges it is told about and learns nothing else — it holds no graph, no model, and never deserializes a payload. That is also why the same relay process serves the [DFL examples](../dfl): it never looks inside what it carries.

The relay stops once every site it has heard from has finished, counting a site from its first publish *or* its first collect. Ctrl-C stops it at any point.

## The Suzuki–Miyaura study

Four laboratories optimizing a Suzuki–Miyaura coupling over 3,696 measured reactions, trading one knowledge token per neighbour per round — the experiment from the ADKO paper's `scientific_discovery` study. Hyperparameters follow its LM-free arm (`b2 l4 g32 s0p5 t50 tb40 a0p01`), and the oracle is the real `suzuki_edbo` yield table rather than a synthetic objective, so a run is a reproduction rather than a demonstration.

There is nothing to download. The dataset ships inside Olympus, as `src/olympus/datasets/dataset_suzuki_edbo/` — 3,696 rows of five categorical parameters and a measured yield, from Perera et al., *Science* **359**, 429–434 (2018) — and `resources/evaluator/suzuki_lookup.py` reads it through `olympus.datasets.load_dataset`, which is also how the reference reads it.

```bash
pip install botorch gpytorch
git clone https://github.com/the-matter-lab/olympus.git ~/olympus
pip install -e ~/olympus
```

Olympus imports its own plotting module at package load, and that module calls `plt.register_cmap`, which matplotlib removed in 3.9 — so on a current matplotlib the first `import olympus` dies with `AttributeError: module 'matplotlib.pyplot' has no attribute 'register_cmap'`. Either pin `matplotlib<3.9` as the reference's `requirements.txt` does, or replace the two `plt.register_cmap(cmap=X)` calls in `src/olympus/plotter/olympus_colors.py` with `matplotlib.colormaps.register(X)`, the supported route since 3.6. Nothing in this example plots, so either is fine.

Olympus' core requirements are otherwise light — numpy, pandas, scikit-learn, matplotlib, seaborn, and SQLAlchemy at an exact `==1.4.45` that will downgrade a 2.x install — because only its dataset half is wanted here. The TensorFlow stack the reference pins is for Olympus' own planners and loads lazily, so it is never reached: ADKO brings its own surrogate.

Two arms, separated by one flag:

- **HET** (`space_kwargs.iid_mode: false`) — each agent may use exactly one of the four solvents. A neighbour's token is then the only route to another solvent's chemistry: with the exchange switched off, three quarters of the grid is unreachable to any one agent. This is the arm the study's claim rests on, and it needs **exactly four agents** — the space refuses any other count, so `--num_nodes` does not apply to it.
- **IID** (`iid_mode: true`) — every agent may propose any reaction. The control arm, and the reference lowers `lam` from 4 to 1 to match.

```bash
# HET, in one process and then one agent per rank
python decentralized/serial/run_serial.py \
    --config ./decentralized/resources/configs/suzuki/simulation/node_0_het.yaml
mpirun -n 4 python decentralized/mpi/run_mpi.py \
    --config ./decentralized/resources/configs/suzuki/simulation/node_0_het.yaml

# the IID control, at any size
mpirun -n 8 python decentralized/mpi/run_mpi.py \
    --config ./decentralized/resources/configs/suzuki/simulation/node_0_iid.yaml
```

All six configs run the study's **200 rounds**, because the point of this example is reproduction: the published figures are 200-round `system_best` traces, and nothing shorter lands on them. Four agents each refitting a GP after every observation costs minutes rather than the toy example's seconds, so every launcher takes `--num_rounds` — `--num_rounds 10` is a smoke run. Shortening it is a faster look at the machinery, not a reproduction.

HET also has a deployed form, because four labs holding four different solvents is the real experiment rather than a simulation of one:

```bash
python decentralized/grpc/run_site.py --config ./decentralized/resources/configs/suzuki/distributed/node_0.yaml
python decentralized/grpc/run_site.py --config ./decentralized/resources/configs/suzuki/distributed/node_1.yaml
python decentralized/grpc/run_site.py --config ./decentralized/resources/configs/suzuki/distributed/node_2.yaml
python decentralized/grpc/run_site.py --config ./decentralized/resources/configs/suzuki/distributed/node_3.yaml
```

A lab that cannot accept inbound connections publishes through `run_relay.py` exactly as in the mixed toy example — point its `grpc_configs` at the relay instead of giving it a `server_uri`, and its peers' `recv_from` entries at the relay for it. Nothing about the experiment changes.

## The CIFAR-10 architecture search

Five agents searching a ResNet design space for CIFAR-10, each training candidates on its own data shard and trading one knowledge token per neighbour per round — the experiment from the ADKO paper's `cifar10` study. Hyperparameters follow its stored run configs (`cifar10/results/{noniid,iid}/config.json`: T=30, B=40, β=2, λ=2, σ_s=0.5, α_τ=0.05, 50 candidates per round), and the oracle is real training rather than a lookup, so a run is a reproduction rather than a demonstration.

A design point θ ∈ `[0, 1]^7` decodes to a CIFAR-adapted ResNet — base width, blocks per group — and its training recipe — learning rate, weight decay, dropout, batch size, cosine schedule. Evaluating it trains that network from scratch for 40 epochs with Adam on the agent's 8,000 training images and returns accuracy on its 2,000 validation images. Every agent searches the same space; what each one owns is its data, so the split lives in `resources/evaluator/cifar10_nas.py` rather than in the space — the reverse of Suzuki.

Nothing beyond APPFL's own install is needed: torch, scikit-learn and OmegaConf are core dependencies, torchvision comes with the `examples` extra. CIFAR-10 is downloaded by torchvision into `datasets/RawData` on first use. On a cluster whose compute nodes are offline, fetch it once from a login node:

```bash
python -c "import torchvision; torchvision.datasets.CIFAR10('./datasets/RawData', train=True, download=True)"
```

Two arms, separated by `evaluator_kwargs.distribution`:

- **non-IID** (`distribution: non_iid`) — each agent's shard is 80% its own block of `10 / num_agents` classes and 20% the rest, and the shards are disjoint. A neighbour's token is then evidence about data the agent never sees. This is the arm the study's claim rests on, with τ=0.7 and γ=32 on a fully connected graph. The class split needs a federation size that divides 10 and fits the shard size, so 2 or 5 agents.
- **IID** (`distribution: iid`) — every agent draws an independent uniform sample. The control arm; the reference runs it with τ=0.5, γ=64, on a random geometric graph of radius 0.6.

```bash
# non-IID, in one process and then one agent per rank
python decentralized/serial/run_serial.py \
    --config ./decentralized/resources/configs/cifar10/simulation/node_0_noniid.yaml
mpirun -n 5 python decentralized/mpi/run_mpi.py \
    --config ./decentralized/resources/configs/cifar10/simulation/node_0_noniid.yaml

# the IID control
mpirun -n 5 python decentralized/mpi/run_mpi.py \
    --config ./decentralized/resources/configs/cifar10/simulation/node_0_iid.yaml
```

Shards are drawn from the training labels with the reference's own random draws, so for the same `seed` each agent trains on exactly the images it does in the reference. The evaluator takes its agent index from `space_kwargs` by interpolation, which is where every launcher writes it. `device: auto` places agent `i` on GPU `i mod (visible GPUs)`, so an MPI run with one GPU per rank uses each rank's own.

Every evaluation is a full training run, so a 30-round run is 150 of them: hours on one GPU, not minutes. The serial launcher trains the five agents one after another; MPI trains them side by side. For a smoke run, set `evaluator_kwargs.n_epochs` to 1 in a copy of the config and pass `--num_rounds 4` — a check of the machinery, not a reproduction.

`evaluator_kwargs.train_seed` seeds every training run from `(train_seed, agent_index, evaluation number)`, so an agent's k-th run starts from the same weights and sees the same batches under every launcher. Set it to `null` to leave torch unseeded.

The non-IID arm also has a deployed form, five sites on a fully connected graph. Each site replays the same global split and keeps its own shard, so `seed`, `num_agents` and the shard settings must match across the five files; only `node_id`, `node_index`, `neighbors`, `space_kwargs.agent_index` and `server_uri` differ:

```bash
python decentralized/grpc/run_site.py --config ./decentralized/resources/configs/cifar10/distributed/node_0.yaml
python decentralized/grpc/run_site.py --config ./decentralized/resources/configs/cifar10/distributed/node_1.yaml
python decentralized/grpc/run_site.py --config ./decentralized/resources/configs/cifar10/distributed/node_2.yaml
python decentralized/grpc/run_site.py --config ./decentralized/resources/configs/cifar10/distributed/node_3.yaml
python decentralized/grpc/run_site.py --config ./decentralized/resources/configs/cifar10/distributed/node_4.yaml
```

A round lasts as long as the slowest site's training run, and a site waits at most an hour for a peer's token, so run these on GPUs.

Three things keep a run from matching the reference exactly, all on the algorithm side rather than in these files. The agent keeps its own tokens in memory and weights every source by 1/(|N|+1), where the reference's CIFAR-10 code keeps only neighbours' tokens at 1/|N|. The reference's published runs blur outgoing token locations with Gaussian noise (σ_φ = 0.05); `ADKOAgent` has an `embedding_privatizer` hook for it, but no config reaches it yet, so these configs send locations un-noised. And GPU training is not bit-for-bit repeatable, so two runs of the same seed agree closely rather than exactly.

## The equivalence this is anchored to

Serial is the control run. The toy 1-D experiment produces the same results under all four paths, and the same results as the original `daisy/prototype` implementation:

```text
serial            vs daisy/prototype   identical   (40x4, 40x8, 15x6 rounds/nodes)
mpi               vs daisy/prototype   identical   (5, 20, 40 rounds)
grpc peer-to-peer vs mpi               identical   (per-site rows, and total bits)
grpc mixed relay  vs mpi               identical   (2 sites served, 2 relayed)
```

The traffic numbers matching is the stronger claim: four independent gRPC processes, each metering only its own sends, sum to the MPI total exactly. Budget enforcement and metering live in `DeOptNodeAgent._payload_to_publish`, which every transport passes through, rather than in each communicator — which is why the accounting survives distribution.

## Configuration

One file per node, and that file is complete. Centralized FL splits configuration in two because a server holds what the clients must agree on; a decentralized federation has no server, so a node's config describes everything it needs — who it talks to, what it optimizes, over what space.

```text
resources/configs/
├── toy1d/
│   ├── simulation/node_0.yaml        from_topo; serial and mpi derive the other nodes from it
│   ├── distributed/node_{0..3}.yaml  explicit; every site serves
│   └── relay/
│       ├── node_{0..3}.yaml          explicit; sites 0,1 serve, sites 2,3 publish to the relay
│       └── relay.yaml                the relay's own endpoint and limits
├── suzuki/
│   ├── simulation/node_0_het.yaml    from_topo; one solvent per agent, exactly four agents
│   ├── simulation/node_0_iid.yaml    from_topo; the control arm, at any size
│   └── distributed/node_{0..3}.yaml  explicit; the HET arm as four deployed labs
└── cifar10/
    ├── simulation/node_0_noniid.yaml from_topo; a disjoint class-skewed shard per agent, 2 or 5 agents
    ├── simulation/node_0_iid.yaml    from_topo; the control arm, at any size
    └── distributed/node_{0..4}.yaml  explicit; the non-IID arm as five deployed sites
```

The algorithm blocks are identical across all of them; only `node_id`, `node_index`, `neighbors` and `comm_configs` differ. `node_index` is a site's position in the federation: it sets the RNG seed offset, and in a heterogeneous experiment it also selects that site's slice of the design space — so the ids are part of the experiment, not labels.

### Resources a config names

`space_configs`, `surrogate_configs` and `evaluator_configs` each name a file and a factory function in `resources/`. They are the experiment, and they are ordinary Python:

```text
resources/
├── space/                              the agent's slice of the design space -- its "data"
│   ├── toy1d_space.py                  an interval of the real line
│   ├── suzuki_space.py                 the 3,696-reaction grid, and the solvent split
│   └── cifar10_nas_space.py            [0, 1]^7, decoded to a ResNet and its training recipe
├── surrogate/                          the model the agent fits to its own measurements
│   ├── kernel_surrogate.py             a kernel-weighted estimator
│   ├── categorical_gp_surrogate.py     a BoTorch GP over categorical inputs
│   └── matern_gp_surrogate.py          a scikit-learn Matern-5/2 GP over continuous inputs
├── evaluator/                          the oracle; ADKO Algorithm 1 step 9
│   ├── toy_yield.py                    a closed-form yield curve
│   ├── suzuki_lookup.py                the measured `suzuki_edbo` yield table
│   └── cifar10_nas.py                  train the decoded ResNet on this agent's CIFAR-10 shard
├── llm_cli.py                          optional LM flags, shared by every launcher
└── reporting.py                        the summary block each launcher prints
```

Swapping experiments is swapping those three paths; nothing in `appfl.decentralized` or in any launcher knows which experiment is running.
