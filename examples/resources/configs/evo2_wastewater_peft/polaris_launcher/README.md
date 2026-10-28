# Polaris client launcher

## Paths to configure before running

- `../client_polaris.yaml`: `trainer_path`
- `pbs_appfl_client.sh`: the `run_client_evo2.py` script path and `--config` path on the final command line
- `launch_client.sh`: the `run_client_evo2.py` script path and `--config` path on the final command line

## Env

The conda env is already built and shared at `/lus/grand/projects/GeomicVar/envs/db_forge_evo2`, usable by anyone in the GeomicVar group. Both `pbs_appfl_client.sh` and `launch_client.sh` already point at it, so there is nothing to set up.

`setup_polaris_environment.sh` is the script that built it. Keep it around in case the env ever needs to be rebuilt.

Before running either launcher, edit `trainer_path` in `../client_polaris.yaml` to point at your own APPFL checkout.

## Batch job

```bash
cd /home/<your-username>/APPFL-evo2-wastewater/examples/resources/configs/evo2_wastewater_peft/polaris_launcher
qsub pbs_appfl_client.sh
```

## Interactive job

1. Get an interactive job

```bash
qsub -I -l select=1:system=polaris:ncpus=32:ngpus=4 \
     -l walltime=01:00:00 -l filesystems=grand:home \
     -q debug -A GeomicVar
```

2. Then launch

```bash
cd /home/<your-username>/APPFL-evo2-wastewater/examples/resources/configs/evo2_wastewater_peft/polaris_launcher
bash launch_client.sh
```
