# Decentralized optimization on APPFL

A federation of agents that each optimize over their own slice of a design space and exchange **knowledge tokens** — compact statements about what worked — rather than models or raw data. [ADKO](../../src/appfl/decentralized/algorithm/adko) is the reference algorithm, and [ADKO v2](../../src/appfl/decentralized/algorithm/adko_v2) — the method its v2 study settled on — runs through the same launchers; the infrastructure under both is APPFL's decentralized FL layer, unchanged.

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

All commands are run from `examples`. **Which launcher you run decides the transport; which config you pass decides the experiment** — the algorithm included: a config names it with `algorithm: adko_v2`, and one that names none runs ADKO. Three experiments are provided:

| | `toy1d` | `suzuki` | `suzuki_v2` |
| --- | --- | --- | --- |
| what it is | a synthetic 1-D yield curve | the Suzuki–Miyaura coupling study from the ADKO paper | the same study under ADKO v2 |
| a design point | a float in `[0, 1]` | five category ids — 3,696 measured reactions | the same |
| the oracle | a closed-form function | the measured `suzuki_edbo` lookup table | the same table, vendored beside the evaluator |
| extra dependencies | none | BoTorch, GPyTorch, Olympus | BoTorch, GPyTorch |
| what it is for | checking that the transports agree | reproducing a published result | reproducing the v2 reference bit for bit |

`toy1d` is the default everywhere, because it runs in seconds and is what the cross-transport equivalence below is measured on. The next three sections apply unchanged to `suzuki` — pass its config instead.

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

## ADKO v2: the settled method

[`adko_v2`](../../src/appfl/decentralized/algorithm/adko_v2) is the method the ADKO v2 study settled on (arm `ADKO-v2f-odds-bit`), one configuration everywhere:

- the score `R = mu + 2 sigma + 2 G - 2 Lambda`, with the peer terms rescaled by `match_rms`;
- 8-bit tokens: a result's robust score against its lab's running median, clipped to ±1;
- one token a round, flagged when it is the lab's new best; receivers keep each peer's newest flagged token past FIFO;
- a FIFO memory of 3 rounds, and trust with odds weights `t/(1-t)`.

The `suzuki_v2` configs run it on the Suzuki study as the v2 study ran it: 4 labs, 5 shared warmup rounds plus 95 rounds of the method, HET or IID. Unlike the v1 example, HET runs with any number of labs (lab `i` gets solvent `i mod 4`), and the oracle is the table the v2 reference vendors, so Olympus is not needed:

```bash
python decentralized/serial/run_serial.py \
    --config ./decentralized/resources/configs/suzuki_v2/simulation/node_0_het.yaml \
    --seed 42 --results_path ./output/adko_v2_het_seed42.json
mpirun -n 8 python decentralized/mpi/run_mpi.py \
    --config ./decentralized/resources/configs/suzuki_v2/simulation/node_0_iid.yaml
python decentralized/grpc/run_site.py \
    --config ./decentralized/resources/configs/suzuki_v2/distributed/node_0.yaml   # and 1..3
```

`--seed` overrides the config's run seed. `--results_path` writes the run record in the v2 reference's own JSON schema (`steps`, `calibration`, `trust`, `token_log`, `best_int`, `system_best`), so its analytics read an APPFL run unchanged; MPI writes it from rank 0, and a gRPC site writes its own lab's. The optional L3 planner (language-model proposals) is in the package, `adko_v2/llm/`, and is not wired into these examples yet.

## The equivalence this is anchored to

Serial is the control run. The toy 1-D experiment produces the same results under all four paths, and the same results as the original `daisy/prototype` implementation:

```text
serial            vs daisy/prototype   identical   (40x4, 40x8, 15x6 rounds/nodes)
mpi               vs daisy/prototype   identical   (5, 20, 40 rounds)
grpc peer-to-peer vs mpi               identical   (per-site rows, and total bits)
grpc mixed relay  vs mpi               identical   (2 sites served, 2 relayed)
```

The traffic numbers matching is the stronger claim: four independent gRPC processes, each metering only its own sends, sum to the MPI total exactly. Budget enforcement and metering live in `DeOptNodeAgent._payload_to_publish`, which every transport passes through, rather than in each communicator — which is why the accounting survives distribution.

ADKO v2 is anchored to its own reference, the v2 study's reference implementation. The serial launcher's run record was compared with the reference's, field by field with `==`: every lab's pick and yield, all four score terms at the pick, the round's calibration, every trust value and every token:

```text
adko_v2 serial vs the v2 reference   bit-identical   HET and IID, seeds 42, 43, 44, 82 (100 rounds)
                                     bit-identical   trust off; a ring; 8 labs (60 rounds)
```

One caveat, which belongs to the platform rather than to either implementation: on macOS, Apple Accelerate's float64 triangular solve with a single right-hand side is not bit-reproducible from one process to the next once the matrix reaches 64 rows, and GP fitting uses it. So from a lab's 64th observation on, the reference does not reproduce *itself* bit for bit (GP hyperparameters move in the 13th digit; in these runs no pick changed), and this implementation differs from it by the same amount. The comparison above ran both sides with that one solve made reproducible — done as a two-column solve, identically in both processes — and changed nothing in either implementation. Single-threaded BLAS (`OMP_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`) is assumed throughout, as the reference's own run scripts set it.

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
└── suzuki_v2/                        `algorithm: adko_v2`, the settled v2 method
    ├── simulation/node_0_het.yaml    from_topo; lab i runs solvent i mod 4, at any size
    ├── simulation/node_0_iid.yaml    from_topo; every lab may run every reaction
    └── distributed/node_{0..3}.yaml  explicit; the HET arm as four deployed labs
```

Within an experiment the algorithm blocks are identical across all of them; only `node_id`, `node_index`, `neighbors` and `comm_configs` differ. `node_index` is a site's position in the federation: it sets the RNG seed offset, and in a heterogeneous experiment it also selects that site's slice of the design space — so the ids are part of the experiment, not labels. A top-level `algorithm` key names the algorithm the launchers build (`adko`, the default, or `adko_v2`); `resources/algorithms.py` is the one place that maps it to a package.

### Resources a config names

`space_configs`, `surrogate_configs` and `evaluator_configs` each name a file and a factory function in `resources/`. They are the experiment, and they are ordinary Python:

```text
resources/
├── space/                              the agent's slice of the design space -- its "data"
│   ├── toy1d_space.py                  an interval of the real line
│   ├── suzuki_space.py                 the 3,696-reaction grid, and the solvent split
│   └── suzuki_v2_space.py              the same for ADKO v2: token noise, the shared warmup
├── surrogate/                          the model the agent fits to its own measurements
│   ├── kernel_surrogate.py             a kernel-weighted estimator
│   ├── categorical_gp_surrogate.py     a BoTorch GP over categorical inputs
│   └── categorical_gp_v2.py            the same GP, standardized, as ADKO v2 reads it
├── evaluator/                          the oracle; ADKO Algorithm 1 step 9
│   ├── toy_yield.py                    a closed-form yield curve
│   ├── suzuki_lookup.py                the measured `suzuki_edbo` yield table, via Olympus
│   ├── suzuki_table.py                 the same table, read from the file beside it
│   └── suzuki_edbo.csv                 3,696 rows; Olympus' data (MIT), as the v2 reference vendors it
├── algorithms.py                       maps a config's `algorithm` to a package
├── llm_cli.py                          optional LM flags, shared by every launcher
├── reporting.py                        the summary block each launcher prints
└── run_record.py                       ADKO v2 run records, in the reference's JSON schema
```

Swapping experiments is swapping those three paths; nothing in `appfl.decentralized` or in any launcher knows which experiment is running.
