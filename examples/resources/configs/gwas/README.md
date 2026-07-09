# Federated GWAS meta-analysis (APPFL / gRPC)

A multi-site GWAS experiment built on **stock APPFL**. Each site runs a local
GWAS on its own genotype/phenotype data; the server combines the per-site
summary statistics with an inverse-variance meta-analysis and emits the pooled
results plus Manhattan/QQ plots. No APPFL core code is modified — only the
example files in this directory.

The pieces specific to this experiment:

| File | Role |
|------|------|
| `site_gwas_trainer.py` | APPFL `Trainer` — runs the local GWAS at each site |
| `meta_analysis_aggregator.py` | APPFL `Aggregator` — meta-analyzes site summary stats |
| `gwas_plot_utils.py` | Manhattan / QQ plotting helpers |
| `gwas_config.py` | shared model factories + config knobs |
| `site_data.py` | loads each site's PLINK + phenotype files |
| `gwas_env.env` | runtime knobs (see below) |
| `server.yaml`, `client_1..3.yaml` | FL configs (one client per site) |

The run scripts are **stock APPFL**: `examples/grpc/run_server.py` and
`examples/grpc/run_client.py`.

## 0. Setup (once per machine)

```bash
# from the repo root
pip install -e .
```

> **Run everything from the repository root.** All paths in these configs are
> relative to the repo root (e.g. `examples/resources/configs/gwas/...`), not to
> `examples/`.

## 1. Drop your data in `local/`

`local/` is git-ignored — your data never gets committed.

**Each site** needs these files under `local/sites/Site<N>/data/`:

```
EUR.synthetic.100k.ld.maf.bed      # PLINK genotype fileset
EUR.synthetic.100k.ld.maf.bim
EUR.synthetic.100k.ld.maf.fam
phenotypes_gwas.csv                # phenotypes for the GWAS
phenotypes_pgs_eval.csv            # phenotypes for PGS evaluation
covariates.csv                     # covariates
```

**The server** needs the variant map for the meta-analysis:

```
local/data_sim/input/EUR.synthetic.100k.ld.maf.bim
```

(`site_data.py` validates these and fails fast with a clear message if any are missing.)

## 2. Edit your client config

In your `client_<N>.yaml`, set:

- `client_id` — your site label (e.g. `Site1`)
- `data_configs.dataset_kwargs.data_dir` — path to your site's data dir
- `comm_configs.grpc_configs.server_uri` — the **server's reachable host:port**

The configs ship pointing at `127.0.0.1:50051` (single-host loopback) for local
testing. For a real multi-institution run, point every client's `server_uri` at
the hosting institution's address, and set the server's `server_uri` (in
`server.yaml`) to bind appropriately (e.g. `0.0.0.0:50051`).

`server.yaml` has `num_clients: 3` — set it to the number of participating sites.

## 3. Run

Start the server (at the hosting institution):

```bash
python examples/grpc/run_server.py \
  --config examples/resources/configs/gwas/server.yaml
```

Then each site starts its client:

```bash
python examples/grpc/run_client.py \
  --config examples/resources/configs/gwas/client_1.yaml
```

Output directories are created automatically.

## 4. Runtime knobs (`gwas_env.env`)

| Key | Meaning |
|-----|---------|
| `Use_cuML` | `true` to use cuML (GPU) regressions instead of scikit-learn |
| `Variant_Scaling` | fraction of variants used at analysis time (`1.0` = all; lower = faster smoke test) |
| `Hit_P_Threshold` | genome-wide significance threshold (default `5e-8`) |
| `Data_Sim_Scaling` | fraction of variants retained at the data-simulation stage |

## 5. Outputs

- **Per site:** `local/sites/Site<N>/output/{data,graphs,logs}`
- **Server (pooled meta-analysis + plots):** `local/server_output/{data,graphs,logs}`
