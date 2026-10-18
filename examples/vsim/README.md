# Virtual-Time FL Simulation

This example runs APPFL clients one at a time on a single CPU or GPU while
reconstructing asynchronous or synchronous target execution with a virtual
clock. It reuses APPFL's `ClientAgent`, `ServerAgent`, scheduler, aggregator,
and trainer; the simulator controls only dispatch, virtual completion, and
aggregation order. The same engine also runs on many GPUs across nodes with MPI;
see [Multi-GPU and multi-node runs](#multi-gpu-and-multi-node-runs-mpi).

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
- `sync_window` aggregates clients that finish within a virtual-time window.
  The clock always advances the full window when the round succeeds, since the
  server cannot know who else is coming until the window closes. If fewer than
  `min_responses` arrive, the round extends up to `max_wait_time` and releases
  the moment the quorum is reached; if the quorum is still unreachable, or no
  `max_wait_time` is set, the round is skipped.

For each client, the basic duration is

```text
compute = seconds_per_step * local_steps * compute_factor
communication = 2 * wire_bytes * 8 / (bandwidth * 1e6)
duration = compute + communication
```

`bandwidth` is Mbps in the standard networking sense — 10^6 bits per second.
`wire_bytes` is the raw model size scaled by `compression_ratio`.

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
                                # (calibrated once at startup with --executor mpi)
    compression_ratio: 1.0      # fraction of the model actually sent (1.0 = raw)
    eval_every: 0               # 0 disables global validation
    task_seeding: true          # seed each local-training task (see below)
    flat_aggregation: true      # FedAsync/FedAvg on a packed global model (see below)
    executor: inprocess         # inprocess | mpi (--executor overrides)
    mpi:                        # used with --executor mpi
      calibration_runs: 2       # train() calls per worker; the last one is timed
      warmup_runs: 1            # with base_step_time: train() calls per worker
                                # before the run (0 = none; see below)
      prefetch_mb: 1024         # results fetched ahead of need, rank 0 budget
      stall_timeout: 1800       # seconds without progress before aborting
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
`--num_local_steps`, `--eval_every`, `--executor`, `--save_history`, and
`--verify`. Run `python vsim/run_vsim.py --help` for the complete list.

`task_seeding` seeds the global RNGs around every local-training task from
(seed, client, how many times the client was selected) and restores them
afterwards. A client's batch order therefore does not depend on how many clients
trained before it in the same process or on which GPU it ran, so the in-process
engine and an MPI run produce the same trajectory. Set it to `false` to keep the
unseeded behavior of earlier releases.

`flat_aggregation` applies to `FedAsyncAggregator` with `AsyncScheduler` and to
`FedAvgAggregator` with `SyncScheduler`, with float32 parameters. The drivers then
keep the global model as one packed buffer and aggregate into it with a few
whole-buffer operations, instead of calling the server agent per update. These are
the aggregators' own operations in the same order, so models and histories are
bit-identical to the agent's path, and the server agent holds the final model
after the run. With `--executor mpi` the packed model is sent as is, and sync
rounds add each model as it arrives rather than after the barrier. Other
aggregators always go through the agent.

To replay a fixed heterogeneity setup rather than sampling one, pass
`--profiles_json` a file of explicit per-client values:

```json
{"Client1": {"compute_factor": 1.81, "bandwidth": 412.7},
 "Client2": {"compute_factor": 0.94, "bandwidth": 248.4}}
```

Clients may be keyed by id or by position, and `delay` is accepted as a synonym
for `compute_factor`.

`compute_factor` is a multiplicative slowdown. `bandwidth` is symmetric Mbps
and communication includes one model download and one upload. The simulator
measures raw tensor bytes and cannot know what a compressor would achieve
without running it, so if `enable_compression` is set in `comm_configs`, declare
the expected ratio via `compression_ratio`; otherwise transfers are modelled
uncompressed and the runner warns. A fixed seed and
fixed `base_step_time` make the virtual schedule reproducible; measured trainer
time can vary with hardware load.

## Outputs and verification

The runner writes console output and `vsim_logs/run_<timestamp>/vsim.log`.
`driver.history` contains one record per async completion or sync round, and
`--save_history out.json` writes it to a file.
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

The sync modes carry one extra requirement. A barrier admits a varying subset of
the dispatched clients each round, so the scheduler must accept a per-round
response count via `set_num_clients()` — `SyncScheduler` does. A scheduler
without it is rejected when the driver is constructed.

## Multi-GPU and multi-node runs (MPI)

With `--executor mpi` the runner splits into one coordinator and many workers.
Rank 0 keeps the virtual clock, the event queue, the barrier logic, and the APPFL
`ServerAgent`. Every other rank is a worker that hosts a share of the virtual
clients and trains them on one device. Launch one worker rank per GPU, plus rank
0, which needs no GPU:

```bash
cd examples
# 1 coordinator + 4 workers on one machine
mpiexec -n 5 python vsim/run_vsim.py --executor mpi --device cuda \
  --server_config vsim/config_vsim_cifar_sync_fedavg.yaml \
  --client_config resources/configs/cifar10/client_1.yaml --num_clients 100 --verify

# PBS, 4 GPUs per node: rank 0 takes one slot, so one GPU on the first node idles
mpiexec -n $((NNODES * 4)) --ppn 4 python vsim/run_vsim.py --executor mpi \
  --device cuda --num_clients 1000 --max_in_flight 128 --save_history h.json
```

Workers on a node are mapped to its visible GPUs round-robin. If the launcher
pins `CUDA_VISIBLE_DEVICES` to one GPU per rank, each worker uses `cuda:0`. More
ranks than GPUs puts several workers on one GPU. On CPU, each worker uses one
torch thread unless `OMP_NUM_THREADS` is set; torch would otherwise start one
thread per core in every rank, which made an 8-worker run 1.85x slower. Worker
output goes to `vsim_logs/run_<timestamp>/worker_<rank>.log`, and per-client log
files are turned off on workers (each `ClientAgent` would otherwise hold four
open files).

With a fixed `base_step_time`, each worker also trains a throwaway client
`warmup_runs` times (default 1) before the run starts, on the thread that trains
its tasks. CUDA and cuDNN start-up then happens during setup rather than in each
worker's first task: in 32-GPU ResNet-18 runs on freshly allocated nodes, first
tasks took 0.3 s with the warm-up and up to 4 s without, against 0.25 s for later
tasks. The throwaway client trains under a seed of its own and the global RNG
states are restored afterwards, so models and histories are the same with or
without the warm-up; `warmup_runs: 0` skips it. Calibration needs no warm-up,
since it trains a throwaway client anyway.

**What moves, and how often.** The event loop never waits on physical training to
order events, and model traffic stays off the critical path:

- Every virtual duration is known at dispatch. Step counts come from a census the
  workers take at startup. The per-step time is either `base_step_time` or
  calibrated once: each worker times a throwaway client (workers on a node take
  turns, so measurements are uncontended), rank 0 uses the median, and it prints
  the value so `--base_step_time` can pin it. Per-task measured durations are not
  supported with MPI: they would serialize the async event loop and depend on
  which GPU ran the task.
- Each client lives on one worker, so its state never moves. Every task costs a
  worker about the same physical time, but fast virtual clients are dispatched
  (async) or accepted (sync) more often, so the runner places clients by their
  expected duration, estimated from the profiles and the configured step count.
  Async runs pack each client's expected update rate onto the least-loaded worker,
  largest first; sync runs deal clients round-robin in order of duration, so the
  clients a barrier accepts spread evenly. In epoch mode, where step counts
  depend on each client's data, client `i` lives on worker `1 + i % W`.
- Workers train in order of virtual completion time, so the result the event loop
  needs next is trained first. A training thread runs alongside a thread that
  serves transfers.
- An async run ends at its `num_global_epochs`-th arrival. With `r` arrivals
  left, that arrival comes no later than the `r`-th earliest pending completion,
  because later dispatches can only add arrivals. Tasks due after it are never
  consumed, so they are not sent, or are dropped on their workers.
- Models are sent as raw byte buffers, never pickled. A global model version goes
  to a worker at most once: sync rounds broadcast it once (`Ibcast`); async
  dispatches send it only to a worker that lacks it. Local models are fetched in
  the order the event loop will consume them, up to `prefetch_mb` ahead.
- In sync modes the barrier is decided before training, so clients the barrier
  would discard are never trained or sent.
- Datasets never move: each worker builds its own clients from the shared
  configuration, and APPFL's partitions are deterministic.

**Reproducibility.** With a fixed `base_step_time`, an MPI run produces the same
schedule as the in-process engine: event order, staleness, epochs, and accepted
clients. On CPU, with the same `OMP_NUM_THREADS`, the models are bit-identical
too. To check a setup:

```bash
export OMP_NUM_THREADS=1
A="--server_config vsim/config_vsim_fedasync.yaml --num_clients 8 --max_in_flight 4 \
   --base_step_time 0.01 --num_global_epochs 30 --verify"
python vsim/run_vsim.py $A --save_history a.json
mpiexec -n 3 python vsim/run_vsim.py --executor mpi $A --save_history b.json
python -c "import json; assert json.load(open('a.json')) == json.load(open('b.json'))"
```

Because discarded clients are not trained, their own bookkeeping can differ from
the in-process engine, which trains every selected client. This affects
VanillaTrainer's round counter, which decides when first-round pre-validation
runs, and differential-privacy accounting. Global models match exactly when there
is no over-selection or `do_pre_validation` is off.

**Sizing.** In async mode each in-flight client holds one model copy on its
worker, about `ceil(max_in_flight / W)` copies per worker; the runner prints the
estimate. Keep `max_in_flight` at four or more times the number of workers, or
workers idle while their clients are not in flight. Rank 0 holds the global
model, at most `prefetch_mb` of fetched local models, and in sync modes the
round's accepted models, as APPFL's `SyncScheduler` always does (with
`flat_aggregation`, one running sum instead). The run ends
with an execution summary: setup and run time, worker utilization, time rank 0
spent waiting, tasks trained, used and dropped, model sends and sends avoided,
broadcasts, and megabytes moved.

**Requirements and limits.**

- The trainer's step count must be known before training: VanillaTrainer in step
  or epoch mode, or a client exposing `expected_local_steps()`. Poisson-sampled
  differential-privacy batches are rejected.
- Local models must be plain tensor state dicts (compression is not supported).
- Each model must be smaller than 2 GiB.
- Library use: build `appfl.vsim.mpi.MPIExecutor` on rank 0, pass it to a driver
  as `executor=`, and call `appfl.vsim.mpi.run_worker` on every other rank. Pass
  the executor `durations`, `mode` and `max_in_flight` to balance homes, and the
  async driver `target_epochs` when its scheduler counts one epoch per arrival
  (as `AsyncScheduler` does), so unconsumable tasks are dropped. See
  `run_vsim.py`.

## Scope

This engine does not model directional or congested links, dynamic availability,
or timeout policies. Those capabilities should be added as separate extensions
without changing the virtual-time ordering contract defined here.
