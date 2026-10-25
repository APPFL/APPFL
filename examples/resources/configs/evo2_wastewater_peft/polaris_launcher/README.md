# Polaris client launcher

## First-time env setup 

Build your own with:

```bash
qsub -I -l select=1:system=polaris:ncpus=32:ngpus=1 \
     -l walltime=01:00:00 -l filesystems=grand \
     -q debug -A GeomicVar

bash setup_polaris_environment.sh /lus/grand/projects/GeomicVar/<your-username>/db-forge
```

This creates the env at `/lus/grand/projects/GeomicVar/<your-username>/db-forge/envs/db_forge_evo2`.
Edit the `ENV=` line in both `pbs_appfl_client.sh` and `launch_client.sh` to that path before running them.

Inside `./resources/configs/evo2_wastewater_peft/client_polaris.yaml` edit the `trainer_path` to your username

## Batch job

```bash
cd /home/<your-username>/APPFL/examples/resources/configs/evo2_wastewater_peft/polaris_launcher
qsub pbs_appfl_client.sh
```

## Interactive job

1. First get an interactive job

```bash
qsub -I -l select=1:system=polaris:ncpus=32:ngpus=4 \
     -l walltime=01:00:00 -l filesystems=grand:home \
     -q debug -A GeomicVar
```

2. Then launch

```bash
cd /home/<your-username>/APPFL/examples/resources/configs/evo2_wastewater_peft/polaris_launcher
bash launch_client.sh
```
