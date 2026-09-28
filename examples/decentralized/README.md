# Decentralized examples

Same shape as the federated-learning examples one directory up: **a launcher per
transport, an experiment per config**. `serial/`, `mpi/` and `grpc/` contain no
science and no hyperparameters -- they construct a
[`TokenExchange`](../../src/appfl/decentralized/exchange) and drive rounds. What is
being optimized lives entirely in `resources/`.

```text
examples/decentralized/
├── serial/run_serial.py      every agent in one process           (the control run)
├── mpi/run_mpi.py            one agent per rank, peer-to-peer     (the HPC path)
├── grpc/run_relay.py         the token relay -- run once
├── grpc/run_site.py          one agent per site, dialling out     (the deployment path)
├── resources/
│   ├── configs/              federation_*.yaml + agent*.yaml, per experiment
│   ├── space/                the design space an agent may probe  (its "data")
│   ├── surrogate/            the model it fits privately          (its "model")
│   ├── evaluator/            the oracle it measures with          (its "loss")
│   ├── llm_cli.py            the --llm_* flags
│   └── reporting.py          the result block, printed identically by all of them
└── reference/                reproduction of the published ADKO Suzuki study
```

## Run it

From this directory (paths in the configs also resolve from the repository root):

```bash
# one process
python serial/run_serial.py

# four MPI ranks, one agent each
mpirun -n 4 python mpi/run_mpi.py

# four sites over gRPC: start the relay, then each site in its own terminal
python grpc/run_relay.py --server_uri localhost:50051
python grpc/run_site.py --agent_id agent-0
python grpc/run_site.py --agent_id agent-1
python grpc/run_site.py --agent_id agent-2
python grpc/run_site.py --agent_id agent-3
```

All three report the same per-agent results, the same token count and the same
total bits -- in the gRPC case, summed across the four sites. If they don't,
distribution changed the algorithm rather than just its plumbing, which is the
bug this set exists to catch.

## Run a different experiment

Nothing above changes. Point the launcher at other configs:

The two flags default independently and both belong to one experiment, so pass
them together. Each config carries an `experiment:` tag and a mismatched pair is
refused at load time rather than quietly running.

```bash
# still the toy objective, with the tuned asymmetric weights (lam 4, gamma 32)
# instead of the v2 defaults -- the weighting changes, the problem does not
python serial/run_serial.py \
    --federation_config ./resources/configs/toy1d/federation_adko_tuned.yaml

# the real Suzuki-Miyaura study, non-IID (needs BoTorch, GPyTorch, Olympus)
python serial/run_serial.py \
    --federation_config ./resources/configs/suzuki/federation_adko_het.yaml \
    --agent_config      ./resources/configs/suzuki/agent_het.yaml

# ... the same experiment across four MPI ranks
mpirun -n 4 python mpi/run_mpi.py \
    --federation_config ./resources/configs/suzuki/federation_adko_het.yaml \
    --agent_config      ./resources/configs/suzuki/agent_het.yaml
```

| Experiment | Federation config | Agent config | What it is | Needs |
| --- | --- | --- | --- | --- |
| `toy1d` | `toy1d/federation_adko.yaml` | `toy1d/agent.yaml` | A closed-form yield curve on `[0, 1]`, four overlapping windows, 40 rounds | nothing beyond `appfl` |
| `toy1d` | `toy1d/federation_adko_tuned.yaml` | `toy1d/agent.yaml` | The same toy objective, weighted the way the Suzuki study weights it | nothing beyond `appfl` |
| `toy1d` | `toy1d/federation_adko_nocomm.yaml` | `toy1d/agent.yaml` | The same toy objective with `lam = gamma = 0`: independent GP-UCB | nothing beyond `appfl` |
| `suzuki` | `suzuki/federation_adko_het.yaml` | `suzuki/agent_het.yaml` | The real `suzuki_edbo` dataset, one solvent per agent, 200 rounds | BoTorch, GPyTorch, Olympus |
| `suzuki` | `suzuki/federation_adko_iid.yaml` | `suzuki/agent_iid.yaml` | The same dataset, all 3,696 reactions open to every agent | BoTorch, GPyTorch, Olympus |

Anything named `toy1d/*` is the one-dimensional stand-in, whatever else is in the
file name. The real chemistry is `suzuki/*` and nowhere else.

## How a config is laid out

Two files, matching the server/client split of the FL examples:

- **`federation_*.yaml`** -- everything every agent must agree on: the graph, the
  round count, the ADKO weights, the surrogate, the communication budget. Two
  sites running different copies of this file are running different experiments.
- **`agent*.yaml`** -- everything local to one site: its slice of the design
  space, its oracle, and how it reaches the relay. This is the only file that
  describes anything private, and it is the analogue of `client_N.yaml`.

The launcher loads one `agent*.yaml` and fills in `agent_id` and
`space_kwargs.agent_index` per agent, exactly as `run_serial.py` fills in
`client_id` and `dataset_kwargs.client_id` for FL.

`resources/space`, `resources/surrogate` and `resources/evaluator` are loaded by
path, like `resources/dataset`, `resources/model` and `resources/loss` in the FL
examples. Each exposes one factory -- `get_design_space`, `get_surrogate`,
`get_evaluator` -- named in the config. **A new experiment is three files and a
config, and no launcher changes.**

Reading those configs is the library's job, not the examples':
[`appfl.decentralized.config`](../../src/appfl/decentralized/config.py) loads the
two files and builds the topology and the budget, and
[`appfl.decentralized.algorithm.adko.create_agent`](../../src/appfl/decentralized/algorithm/adko/builder.py)
builds the agent -- the counterpart of `appfl.agent.ClientAgent` for FL. Every
site in a distributed run therefore constructs its agent through the same code,
which is what makes "the same experiment three ways" checkable rather than
merely intended.

An ablation arm is another `federation_*.yaml`, the way `server_fedprox.yaml` is one for FL.
`federation_adko_nocomm.yaml` is the no-communication baseline -- `lam` and `gamma` set to
zero, which collapses every agent to independent GP-UCB:

```bash
python serial/run_serial.py \
    --federation_config ./resources/configs/toy1d/federation_adko_nocomm.yaml
```

## What `toy1d` shows, and what it does not

`true_yield(x)` is a stand-in for an expensive measurement: a peak of 100 at
`x=0.35` plus a decoy of 40 at `x=0.75`, on a 0-100 yield scale with success
threshold 50. The four agents own overlapping windows of `[0, 1]`:

| agent | window | grid points | best it can reach alone |
| --- | --- | --- | --- |
| agent-0 | [0.00, 0.30] | 16 | 77.9 |
| agent-1 | [0.25, 0.50] | 13 | **99.0** |
| agent-2 | [0.45, 0.75] | 15 | 39.8 |
| agent-3 | [0.70, 1.00] | 16 | 39.8 |

Only agent-1's window contains the peak, which is the shape that makes a
neighbour's token worth anything.

**This objective does not demonstrate that communication helps, and is not meant
to.** Candidates are quantized to a 50-bin grid, so the four windows hold 60
points between them -- and at the default 40 rounds every agent simply enumerates
its whole window and stops. That is why a run reports exactly 60 evaluations and
60 tokens, and why each site's token count equals its grid size. The search is
over by round ~16.

That is fine for what `toy1d` tests, which is whether three transports produce
identical output. For a benchmark where communication has room to matter, use the
`suzuki` configs -- 924 candidates per agent over 200 rounds, on the dataset the
ADKO paper uses.

## `reference/`

A separate driver that reproduces the published ADKO `scientific_discovery`
Suzuki study: it reads the reference's own experiment JSON, replays the reference's
warmup bank, writes results in the reference's schema, and compares the two
seed-by-seed. It uses the same `resources/space/suzuki_space.py` and
`resources/surrogate/categorical_gp_surrogate.py` as the configs above, so the two
paths cannot drift apart. See [`reference/README.md`](reference/README.md).
