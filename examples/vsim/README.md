# Virtual-Time FL Simulation

This example runs APPFL clients one at a time on a single CPU or GPU while
reconstructing asynchronous or synchronous target execution with a virtual
clock. It reuses APPFL's `ClientAgent`, `ServerAgent`, scheduler, aggregator,
and trainer; the simulator controls only dispatch, virtual completion, and
aggregation order.

Physical training order and virtual arrival order are intentionally separate.
The simulator never sleeps to reproduce a delay: it advances directly to the
next virtual event or barrier.

## Modes

- `async` keeps up to `max_in_flight` clients virtually in flight. A min-heap
  orders completion events, and updates reach the APPFL server in virtual
  arrival order. This reconstructs asynchronous completion order and staleness
  although local training is physically serial. Note that every arrival
  aggregates on its own: `max_in_flight` bounds outstanding updates, it is not
  a quorum.
- `sync_count` dispatches `participants_per_round` clients and aggregates the
  first `min_responses` virtual completions. Remaining completions are
  discarded for that round.
- `sync_window` aggregates clients that finish within a virtual-time window. A
  minimum response count and optional hard deadline control whether the round
  is accepted or skipped.

For each client, the basic duration is

```text
compute = seconds_per_step * local_steps * compute_factor
communication = 2 * model_bits / bandwidth
duration = compute + communication
```

By default, `seconds_per_step` comes from APPFL trainer metadata for the actual
local training task. Set `base_step_time` to supply a fixed seconds-per-step
value instead. Compute, communication, availability, timeout, and dedicated
time-source models are outside this basic engine.

## Quick start

Run commands from `examples/` because the YAML files use relative resource
paths. Use `--device cuda` to run serial local training on one GPU.

```bash
cd examples

python vsim/run_vsim.py \
  --server_config vsim/config_vsim_fedasync.yaml \
  --num_clients 10 --device cpu --verify

python vsim/run_vsim.py \
  --server_config vsim/config_vsim_sync_count.yaml \
  --num_clients 10 --mode sync_count --device cpu --verify

python vsim/run_vsim.py \
  --server_config vsim/config_vsim_sync_window.yaml \
  --num_clients 10 --mode sync_window --device cpu --verify
```

## Configuration

Simulator settings are under `server_configs.simulator`. Standard APPFL
scheduler, aggregator, model, trainer, and data settings are unchanged.

```yaml
server_configs:
  simulator:
    seed: 42
    mode: async                 # async | sync_count | sync_window
    max_in_flight: 4            # async: dispatched but not yet arrived
    base_step_time: null        # seconds/step; null uses trainer measurement
    eval_every: 0               # 0 disables global validation
    heterogeneity:
      compute:
        distribution: lognormal
        params: {mu: 0.0, sigma: 0.5}
      bandwidth:
        distribution: uniform
        params: {lo: 150.0, hi: 600.0}  # Mbps
    sync:
      participants_per_round: 10   # dispatched each round
      min_responses: 8             # needed before the round is accepted
      window_duration: 30.0        # seconds; required by sync_window
      max_wait_time: 60.0          # optional hard deadline in seconds
```

Important command-line overrides are `--mode`, `--num_clients`, `--device`,
`--seed`, `--max_in_flight`, `--base_step_time`, `--num_global_epochs`,
`--num_local_steps`, `--eval_every`, and `--verify`. Run
`python vsim/run_vsim.py --help` for the complete list.

`compute_factor` is a multiplicative slowdown. `bandwidth` is symmetric Mbps
and communication includes one model download and one upload. A fixed seed and
fixed `base_step_time` make the virtual schedule reproducible; measured trainer
time can vary with hardware load.

## Outputs and verification

The runner writes console output and `vsim_logs/run_<timestamp>/vsim.log`.
`driver.history` contains one record per async completion or sync round.
`plot_timeline.py` can visualize async history exported by a caller.

`--verify` checks virtual-time monotonicity, completion arithmetic, requested
completion/round counts, concurrency bounds, nonnegative async staleness, and
positive durations. These checks validate engine bookkeeping; they do not claim
that a chosen duration profile matches a particular deployment.

## Using another APPFL algorithm

Select the normal APPFL scheduler and aggregator in the server YAML. No new
simulator implementation is required when the algorithm uses the public agent,
scheduler, and aggregator interfaces. Async algorithms use `async`; synchronous
algorithms use `sync_count` or `sync_window`.

## Scope

This engine physically trains clients serially on one device. It does not model
directional or congested links, dynamic availability, timeout policies, or
multi-GPU execution. Those capabilities should be added as separate extensions
without changing the virtual-time ordering contract defined here.
