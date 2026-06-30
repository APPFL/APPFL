# vsim — Virtual-Time FL Simulator for APPFL

A virtual-time federated learning simulator that runs asynchronous and
synchronous FL on a **single CPU/GPU** while faithfully reproducing client
arrival order, timing, and staleness dynamics. No distributed infrastructure
needed — plug in any APPFL aggregator/scheduler and get virtual-time simulation
with zero simulator code change.

Built **natively on APPFL** — the only additions are the `appfl.simulator`
package and these example files. Zero APPFL core modifications.

## Key Features

- **Dual-mode simulation** — training mode runs real PyTorch training with
  virtual-time ordering; timing-only mode skips training entirely and computes
  durations from profiles (1000+ clients in seconds)
- **Async + sync FL** — async event queue (FedAsync, FedBuff, FedCompass);
  sync with count-barrier and window-barrier modes (FedAvg, FedAvgM, FedAdam, etc.)
- **Realistic system modeling** — 4 compute modes (measured, factor, profile,
  tier) with 16 device profiles; asymmetric bandwidth, jitter, shared BW pool,
  TCP overhead; 3 availability/dropout models + timeout
- **15+ algorithms, zero simulator change** — any APPFL aggregator works
  automatically via the decoupled event engine
- **Framework-native** — uses APPFL's public API only (ServerAgent, ClientAgent,
  schedulers, aggregators). Existing APPFL experiments convert to virtual-time
  simulation by adding a `simulator:` config block
- **Deterministic reproducibility** — fixed seed + `base_step_time` produces
  bit-exact replays
- **HPC-ready** — APPFL provides MPI, Globus Compute, PBS, lossy compressors
  (SZ2/SZ3/ZFP/SZx); all available to the simulator without additional work
- **Timeline visualization** — per-client Gantt charts (compute, communication,
  aggregation) with full and zoomed views

## Comparison with Existing Simulators

| Feature | AFL-Lib | FedDES | FedScale | Plato | Flower | NVFlare | **Ours** |
|---------|:-------:|:------:|:--------:|:-----:|:------:|:-------:|:--------:|
| Virtual-time simulation | ✓ | ✓ | ✓ | △¹ | ✗ | ✗ | **✓** |
| Training mode | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | **✓** |
| Timing-only mode | ✗ | ✓ | △² | ✗ | ✗ | ✗ | **✓** |
| Decoupled event engine | ✗ | ✓ | ✓ | ✗ | — | — | **✓** |
| Compute heterogeneity | △³ | ✓ | ✓ | △⁴ | ✗ | ✗ | **✓** |
| Communication modeling | △⁵ | ✓ | △⁵ | △⁵ | ✗ | ✗ | **✓** |
| Availability modeling | △⁶ | △ | ✓ | ✗ | △⁷ | ✗ | **✓** |
| Async FL | ✓ | ✓ | △⁸ | ✓ | ✗ | △⁹ | **✓** |
| Sync FL | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** |
| HPC-native | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| Framework-native | — | — | — | — | — | — | **✓** |

<details><summary>Footnotes (△ = partial support)</summary>

1. **Plato:** Uses `time.sleep()` for delay — burns real wall-clock time, not virtual-time fast-forward.
2. **FedScale:** Virtual time for scheduling, but still dispatches real training to executors.
3. **AFL-Lib compute:** 3 hardcoded IoT devices × SCALE_FACTOR=50 only.
4. **Plato compute:** Sleep-distribution stragglers, not device-modeled compute time.
5. **Comm modeling (AFL-Lib, FedScale, Plato):** Simple `bytes/bandwidth` formula — no jitter, no asymmetric BW, no congestion.
6. **AFL-Lib availability:** Adds random extra latency, not actual offline/online session modeling.
7. **Flower availability:** `fraction_fit` controls sampling fraction — no dynamic dropout or time-varying patterns.
8. **FedScale async:** Has async aggregator but limited to ~1 async strategy.
9. **NVFlare async:** FedBuff exists only in `edge/` subsystem for cross-device, not in the standard workflow.

</details>

## File Structure

### Simulator core (`src/appfl/simulator/`)

| File | Role |
|------|------|
| `__init__.py` | Package exports |
| `base_sim_driver.py` | Base class — common state, utilities, calibration, verification |
| `async_sim_driver.py` | Async driver — min-heap event queue, train_start/train_complete handlers |
| `sync_sim_driver.py` | Sync driver — count-barrier and window-barrier round logic |
| `client_profile.py` | Per-client heterogeneity profile (compute factor + bandwidth) |
| `compute_model.py` | 4-mode compute model with 16 device profiles |
| `comm_model.py` | Communication model (asymmetric BW, jitter, shared pool, TCP overhead) |
| `availability_model.py` | 3 dispatch-level dropout models + completion-level timeout |

### Example scripts (`examples/vsim/`)

| File | Role |
|------|------|
| `run_vsim.py` | Entry point — config loading, agent creation, driver invocation |
| `plot_timeline.py` | Timeline visualization (Gantt charts) |
| `server_val_mnist.py` | Server-side MNIST test set for global evaluation |
| `server_val_cifar.py` | Server-side CIFAR-10 test set for global evaluation |
| `config_vsim_*.yaml` | Experiment configurations (see below) |

## Quick Start

Run from the `examples/` directory (configs use `./resources/...` relative paths):

```bash
cd examples

# Async FL — FedAsync, 10 clients, K=4 concurrent, MNIST
python vsim/run_vsim.py \
    --server_config vsim/config_vsim_fedasync.yaml \
    --num_clients 10 --seed 42 --verify

# Sync FL — count mode, 10 clients, aggregate first K=8
python vsim/run_vsim.py \
    --server_config vsim/config_vsim_sync_count.yaml \
    --num_clients 10 --mode sync_count --verify

# Timing-only — skip training, profile-based durations
python vsim/run_vsim.py \
    --server_config vsim/config_vsim_fedasync.yaml \
    --num_clients 100 --timing_only --base_step_time 0.003 --seed 42 --verify
```

## CLI Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--server_config` | str | `./vsim/config_vsim_fedasync.yaml` | Path to server/simulator YAML config |
| `--client_config` | str | `./resources/configs/mnist/client_1.yaml` | Base client config (data, trainer, model) |
| `--num_clients` | int | `2` | Number of simulated clients |
| `--seed` | int | config or `42` | RNG seed (overrides `simulator.seed` in config) |
| `--device` | str | `cpu` | PyTorch device (`cpu` or `cuda`) |
| `--base_step_time` | float | `None` | Fixed per-step compute time (s). `None` = use measured GPU time |
| `--mode` | str | `async` | Simulation mode: `async`, `sync_count`, `sync_window` |
| `--timing_only` | flag | `False` | Skip training; compute virtual time from profiles only |
| `--verify` | flag | `False` | Run post-simulation invariant checks |
| `--eval_every` | int | config or `0` | Global-model evaluation every N completions (0 = off) |
| `--max_concurrency` | int | config or N | Max concurrent in-flight clients (K) |
| `--num_global_epochs` | int | config | Override total global update target |
| `--target_epochs` | int | config | Target rounds for timing-only mode |
| `--num_local_steps` | int | config | Override local training steps per client |
| `--partition` | str | `None` | Override data partition strategy (e.g. `dirichlet_noniid`) |
| `--alpha` | float | `None` | Dirichlet alpha (mapped to `alpha2 = alpha × num_classes`) |
| `--num_classes` | int | `10` | Number of classes for alpha scaling |
| `--het_json` | str | `None` | AFL-Lib per-client `(delay, bandwidth)` JSON for exact matching |
| `--afl_dir` | str | `None` | AFL-Lib root for loading npz data shards |
| `--afl_dataset` | str | `None` | AFL dataset directory name (e.g. `mnist10`) |

## Configuration Reference

All simulator settings live under `server_configs.simulator` in the YAML config.
Standard APPFL settings (`scheduler`, `aggregator`, `train_configs`, etc.) are
unchanged — see APPFL documentation.

```yaml
server_configs:
  scheduler: "AsyncScheduler"          # APPFL scheduler (existing)
  aggregator: "FedAsyncAggregator"     # APPFL aggregator (existing)
  num_global_epochs: 100               # total aggregation rounds

  simulator:                           # ← simulator-specific settings
    seed: 42
    max_concurrency: 4                 # K: max in-flight clients (async)
    mode: "async"                      # async | sync_count | sync_window
    staleness_mode: "round"            # staleness counting method
    eval_every: 10                     # global eval frequency (0 = off)

    time_model:
      base_step_time: null             # null = measured GPU time
      mode: null                       # null | fixed | calibration | real_measure
      calibration_epochs: 3            # steps for calibration mode

    heterogeneity:
      compute:
        distribution: "lognormal"      # lognormal | fixed
        params: { mu: 0.0, sigma: 0.5 }
      bandwidth:
        distribution: "uniform"        # uniform | lognormal | fixed
        params: { lo: 150.0, hi: 600.0 }

    compute_model:                     # v2 — advanced compute (optional)
      mode: "measured"                 # measured | factor | profile | tier | flops
      device_types:
        options: ["a100", "rtx3090", "jetson_nano"]
        weights: [1, 2, 1]
      gpu_utilization: 0.5
      model_flops_per_step: 0.0
      tiers:                           # for mode=tier
        - { name: "fast", factor: 1.0, proportion: 1 }
        - { name: "slow", factor: 10.0, proportion: 2 }

    comm_model:                        # v2 — advanced comm (optional)
      download_bw:
        distribution: "uniform"
        params: { lo: 100.0, hi: 500.0 }
      upload_bw:
        distribution: "uniform"
        params: { lo: 50.0, hi: 200.0 }
      jitter_sigma: 0.1               # lognormal jitter (0 = deterministic)
      compression_ratio: 1.0           # 1.0 = no compression
      latency: 0.0                     # base RTT (s) for TCP overhead
      shared_pool:
        total_bandwidth: 1000.0        # Mbps
        mode: "fair_share"             # fair_share | none

    availability:                      # v2 — dropout models (optional)
      mode: "session"                  # none | permanent | session | correlated | composite
      permanent:
        drop_prob: 0.02
      session:
        active_duration: 300.0         # seconds
        inactive_duration: 600.0
        phase_noise: 0.2
      correlated:
        num_groups: 3
        failure_prob: 0.05
        failure_duration: 30.0
      timeout:
        timeout_seconds: 120.0         # or timeout_quantile: 0.95

    sync:                              # sync mode settings
      participants_per_round: 10       # M: clients dispatched per round
      min_responses: 8                 # K: minimum for aggregation
      max_wait_time: 120.0             # hard deadline (seconds)
      window_duration: 60.0            # window mode deadline
```

### Parameter Details

| Parameter | Type | Default | Used by | Description |
|-----------|------|---------|---------|-------------|
| **General** | | | | |
| `seed` | int | `42` | All drivers | RNG seed for reproducibility |
| `max_concurrency` | int | N | `AsyncSimDriver` | K: max concurrent in-flight clients |
| `mode` | str | `async` | `run_vsim.py` | Selects driver: async, sync_count, sync_window |
| `staleness_mode` | str | `round` | `AsyncSimDriver` | Staleness counting method |
| `eval_every` | int | `0` | All drivers | Global evaluation frequency (0 = off) |
| **time_model** | | | | |
| `base_step_time` | float | `None` | All drivers | Per-step compute time (s). None = measured |
| `mode` | str | `None` | `run_vsim.py` | fixed, calibration, real_measure |
| `calibration_epochs` | int | `3` | `BaseSimDriver` | Profile steps before switching to timing-only |
| **heterogeneity** | | | | |
| `compute.distribution` | str | — | `build_profiles()` | lognormal or fixed |
| `compute.params.mu/sigma` | float | 0.0/0.5 | `build_profiles()` | Lognormal parameters |
| `bandwidth.distribution` | str | — | `build_profiles()` | uniform, lognormal, or fixed |
| `bandwidth.params.lo/hi` | float | 150/600 | `build_profiles()` | Bandwidth range (Mbps) |
| **compute_model** | | | | |
| `mode` | str | `measured` | `ComputeModel` | measured, factor, profile, tier, flops |
| `device_types.options` | list | `["a100"]` | `build_compute_models()` | Device pool to sample from |
| `device_types.weights` | list | `[1.0]` | `build_compute_models()` | Sampling weights |
| `gpu_utilization` | float | `0.5` | `ComputeModel` | GPU utilization for flops mode |
| `tiers` | list | — | `build_compute_models()` | Tier definitions (name, factor, proportion) |
| **comm_model** | | | | |
| `download_bw` | dist | 300 Mbps | `build_comm_models()` | Download bandwidth distribution |
| `upload_bw` | dist | = download | `build_comm_models()` | Upload bandwidth distribution |
| `jitter_sigma` | float | `0.0` | `CommModel` | Lognormal jitter (0 = deterministic) |
| `compression_ratio` | float | `1.0` | `CommModel` | Effective compression ratio |
| `latency` | float | `0.0` | `CommModel` | Base RTT seconds (TCP overhead) |
| `shared_pool.total_bandwidth` | float | 1000 | `SharedBandwidthPool` | Total shared bandwidth (Mbps) |
| `shared_pool.mode` | str | `fair_share` | `SharedBandwidthPool` | fair_share or none |
| **availability** | | | | |
| `mode` | str | `none` | `build_availability()` | none, permanent, session, correlated, composite |
| `permanent.drop_prob` | float | `0.02` | `PermanentDropout` | Probability of permanent exit |
| `session.active_duration` | float | `300.0` | `SessionDropout` | Active window (seconds) |
| `session.inactive_duration` | float | `600.0` | `SessionDropout` | Inactive window (seconds) |
| `session.phase_noise` | float | `0.2` | `SessionDropout` | Phase jitter fraction |
| `correlated.num_groups` | int | `3` | `CorrelatedDropout` | Number of failure groups |
| `correlated.failure_prob` | float | `0.05` | `CorrelatedDropout` | Group failure probability |
| `correlated.failure_duration` | float | `30.0` | `CorrelatedDropout` | Failure duration (seconds) |
| `timeout.timeout_seconds` | float | — | `TimeoutModel` | Absolute timeout threshold |
| `timeout.timeout_quantile` | float | — | `TimeoutModel` | Adaptive quantile threshold |
| **sync** | | | | |
| `participants_per_round` | int | N | `SyncSimDriver` | M: clients dispatched per round |
| `min_responses` | int | M | `SyncSimDriver` | K: minimum for aggregation |
| `max_wait_time` | float | — | `SyncSimDriver` | Hard deadline (seconds) |
| `window_duration` | float | — | `SyncSimDriver` | Window mode deadline (seconds) |

## How It Works

**Async mode:** A min-heap event queue holds `(virtual_time, seq, event_type,
client_id)` tuples. The driver dispatches up to K clients, each producing a
`train_start` event. When a client's virtual training completes (compute_time
+ comm_time from its profile), a `train_complete` event is pushed. The main loop
pops the earliest event and jumps virtual time — no wall-clock sleeping. Slow
clients stay in-flight while fast clients aggregate multiple times, producing
realistic staleness dynamics.

**Sync mode:** Each round selects M participants (availability-filtered), trains
all of them sequentially on the physical GPU, then applies a barrier. In
**count mode**, the first K completions (by virtual time) are aggregated; the
rest are discarded. In **window mode**, all completions within `window_duration`
of the round start are aggregated. Both modes support `max_wait_time` (hard
deadline) and `min_responses` (skip round if too few arrive).

**Virtual time formula:** For each client, `duration = compute_time + comm_time`.
Compute time comes from the client's `ComputeModel` (or v1 formula:
`cps × steps × compute_factor`). Communication time comes from `CommModel`
(or v1: `model_bytes × 8 / bandwidth / 1024² × 2`). The `ClientProfile`
dispatches to whichever model is configured.

## Output

- Console + `vsim_logs/run_<timestamp>/vsim.log` — per-event `START`/`DONE`
  lines with virtual time, staleness, per-client accuracy, and (if `eval_every > 0`)
  `GLOBAL` lines with the global-model accuracy curve.
- `driver.history` — list of dicts, one per completion (async) or round (sync):
  - Async: `vtime`, `cid`, `epoch`, `staleness`, `val_accuracy`, `duration`,
    `dispatch_time`, `comm_bytes`, `global_val_accuracy` (if eval)
  - Sync: `round`, `vtime`, `t_start`, `t_barrier`, `accepted_count`,
    `discarded_count`, `accepted_cids`, `duration`, `comm_bytes`
- `--verify` runs post-simulation invariant checks:
  - Async: 6 checks (monotonic time, completion arithmetic, count, concurrency, staleness, positive durations)
  - Sync: 5 checks (monotonic round starts, completed rounds, barrier respected, positive durations, concurrency)

## Determinism

With a fixed `base_step_time` and seed, repeated runs produce bit-exact results.
When using measured GPU time (`base_step_time=None`), virtual durations vary
slightly across runs due to hardware timing noise — use
`time_model.mode=calibration` to measure once and then switch to timing-only.
