# vsim — Virtual-Time Asynchronous FL Simulator (v1)

Run asynchronous federated learning on a **single CPU/GPU** while faithfully
reproducing asynchronous client *arrival order* in **virtual time** — no
distributed execution. This is the asynchronous counterpart of
`examples/serial/run_serial.py` (which is synchronous-only).

It reuses APPFL's `ServerAgent` / `ClientAgent` / scheduler / aggregator
**without modifying APPFL core** — the only additions are the `appfl.simulator`
package and these example files.

## How it works

- A min-heap **event queue** holds `(virtual_time, seq, type, client_id)`.
- Each client trains for real; its **completion time** is modeled as
  `compute + comm`, where
  `compute = compute_second_per_step * num_local_steps * compute_factor`
  (per-step time is measured by APPFL's trainer) and
  `comm = model_bytes * 8 / bandwidth * 2`.
- Virtual time **jumps to the next completion event**. Slow clients stay
  in-flight while fast clients aggregate repeatedly → realistic staleness.
- `max_concurrency` (K) clients are in flight at once (AFL-Lib style).

Staleness semantics follow **APPFL's native FedAsync** (update-count based), so
this reproduces AFL-Lib *dynamics*, not bit-identical numbers.

## Files

| File | Role |
|---|---|
| `src/appfl/simulator/async_sim_driver.py` | `AsyncSimDriver` — event queue + virtual clock |
| `src/appfl/simulator/client_profile.py` | `ClientProfile` — per-client compute_factor + bandwidth |
| `run_vsim.py` | entry point (config → agents → driver) |
| `config_vsim_fedasync.yaml` | base config (FedAsync, 2 clients) |
| `config_vsim_homo.yaml` / `config_vsim_hetero.yaml` | homogeneous vs heterogeneous experiment |
| `config_vsim_scale8.yaml` | 8-client scale-up |
| `config_vsim_globaleval.yaml` | + server-side global-model evaluation |
| `server_val_mnist.py` | server validation set (full MNIST test) for global eval |

## Run

Run from the `examples/` directory (configs use `./resources/...` relative paths):

```bash
cd examples

# basic smoke run (2 clients)
python vsim/run_vsim.py --num_clients 2 --seed 42

# heterogeneous, 4 clients, with global-model accuracy logging
python vsim/run_vsim.py --server_config ./vsim/config_vsim_globaleval.yaml \
    --num_clients 4 --seed 42
```

### CLI options
- `--server_config` : simulator config (default `./vsim/config_vsim_fedasync.yaml`)
- `--client_config` : base client config (default MNIST `client_1.yaml`)
- `--num_clients`   : number of clients
- `--seed`          : RNG seed (overrides config)
- `--device`        : `cpu` or `cuda`
- `--base_step_time`: fixed per-step compute time (s). Set for **fully
  deterministic** virtual time; omit to use the measured (realistic) value.

### Config: `server_configs.simulator` block
```yaml
simulator:
  seed: 42
  max_concurrency: 4        # K clients in flight (>=2 for real async dynamics)
  staleness_mode: "round"   # v1 = APPFL native (update-count)
  eval_every: 4             # global-model eval every N completions (0 = off)
  time_model:
    base_step_time: 0.01    # null = use measured per-step time
  heterogeneity:
    compute:   { distribution: "lognormal", params: { mu: 0.0, sigma: 0.5 } }
    bandwidth: { distribution: "uniform",   params: { lo: 150.0, hi: 600.0 } }
```

## Output

- Console + `vsim_logs/run_<timestamp>/vsim.log`: per-event `START`/`DONE` lines
  with `virtual_time`, `staleness`, per-client `val_acc`, and (if `eval_every>0`)
  `GLOBAL` lines with the global-model accuracy curve.
- `driver.history` holds one record per completion
  (`vtime, cid, epoch, staleness, val_accuracy, [global_val_accuracy]`).

Determinism: with a fixed `base_step_time` and seed, repeated runs are bit-exact.

## v1 scope / limitations

v1 reproduces AFL-Lib's serial virtual-time approach. It does **not** yet include
(planned for v2): time-based staleness, realistic availability/dropout, IoT/HPC
profile presets, communication realism (compression / variable bandwidth /
congestion), and physically-concurrent client execution. See
`virtual_sim/notes/03_afl_lib_limitations.md`.
