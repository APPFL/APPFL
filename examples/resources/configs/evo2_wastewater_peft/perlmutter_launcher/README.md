# Perlmutter client launcher

## Env

The conda env is already built and shared at
`/global/cfs/projectdirs/m4994/common/db-forge-poc/evo2_metagenomics/envs/evo2_fl_env312`,
usable by anyone in the `m4994` group. Both `slurm_appfl_client.sh` and
`launch_client.sh` already point at it, so there is nothing to set up.

`setup_perlmutter_environment.sh` is the script that built it. it installs appfl from
this repo checkout (not PyPI), so rebuild it after pulling changes to
`src/appfl`.

Edit the `SCRATCH_BASE=` and `DBFORGE_ROOT=` lines in both
`slurm_appfl_client.sh` and `launch_client.sh` to your own pscratch paths
before running them.

Inside `./resources/configs/evo2_wastewater_peft/client_perlmutter.yaml` edit
`trainer_path` to point at your own APPFL checkout.

## Batch job


```bash
cd /global/homes/<u>/<your-username>/APPFL/examples/resources/configs/evo2_wastewater_peft/perlmutter_launcher
sbatch slurm_appfl_client.sh
```

## Interactive job

1. First get an interactive GPU node:

```bash
salloc -N 1 -C gpu -q interactive -t 00:30:00 -A <your_nersc_account>
```

2. Then launch:

```bash
cd /global/homes/<u>/<your-username>/APPFL/examples/resources/configs/evo2_wastewater_peft/perlmutter_launcher
bash launch_client.sh
```
