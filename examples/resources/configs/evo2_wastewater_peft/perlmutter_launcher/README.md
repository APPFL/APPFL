# Perlmutter client launcher

## First-time env setup

Build your own with:

```bash
bash setup_perlmutter_environment.sh /pscratch/sd/<u>/<your-username>/dbforge/evo2_metagenomics
```

Compiles flash-attn from source like Polaris does, but doesn't need a GPU
present to do it — runs fine on a login node.

This creates the env at
`/pscratch/sd/<u>/<your-username>/dbforge/evo2_metagenomics/envs/evo2_fl_env312`.
Edit the `EVO2_BASE=`/`DBFORGE_ROOT=` lines in both `slurm_appfl_client.sh`
and `launch_client.sh` to your own paths before running them.

Inside `./resources/configs/evo2_wastewater_peft/client_perlmutter.yaml` edit
`trainer_path` and the `base_data`/`model_path`/`evo2_dir`/`aidrin_csv` paths
to your own username.

## Batch job

```bash
cd /home/<your-username>/APPFL/examples/resources/configs/evo2_wastewater_peft/perlmutter_launcher
sbatch slurm_appfl_client.sh
```

## Interactive job

1. First get an interactive GPU node:

```bash
salloc -N 1 -C gpu -q interactive -t 00:30:00 -A <your_nersc_account>
```

2. Then launch:

```bash
cd /home/<your-username>/APPFL/examples/resources/configs/evo2_wastewater_peft/perlmutter_launcher
bash launch_client.sh
```
