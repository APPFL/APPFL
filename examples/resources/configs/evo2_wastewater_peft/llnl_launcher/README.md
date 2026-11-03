# Running the Evo2 FL Client on LLNL

> **Note**: This is based on a working Perlmutter NERSC build of this same
> pipeline. Module names, paths, and the
> scheduler below are placeholders, confirm each against the actual target
> machine (see each section).
>
> **GPU**: requires NVIDIA, compute capability >= 8.0 (Ampere or newer: A100,
> H100, ...). flash-attn 2.x does not support older GPUs (e.g. V100). 
>
> **Scheduler**: assumes SLURM (`sbatch`/`salloc`). 
## Loading Modules

Confirm real module names for your system (`module avail conda`,
`module avail cuda`, `module avail gcc`), load them, then check with
`module list`. It should look something like this (names/versions will
differ, this is just the shape):

```
1) gcc/12.3.0   2) cuda/12.6.0   3) conda/24.x   4) <site-env modules>
```

Then edit the three `module load` lines marked `# TODO` in
`setup_llnl_environment.sh` to match what you loaded.

## Creating Conda Environment and Installing Dependencies

```bash
git clone -b evo2-wastewater-peft-fedavg https://github.com/APPFL/APPFL.git
cd APPFL/examples/resources/configs/evo2_wastewater_peft/llnl_launcher
bash setup_llnl_environment.sh /p/lustre1/<your-username>/dbforge/evo2_metagenomics
```

The path argument is `<base_dir>`, a directory you pick that doesn't need to
exist yet. The script creates everything under it: the conda env
(`<base_dir>/envs/evo2_fl_env312`), a clone of the evo2 repo
(`<base_dir>/evo2`), and pip/triton caches. This same path is `EVO2_BASE` in
the batch script below.

`setup_llnl_environment.sh` installs, in order:

| Library | Version | Why this order |
|---|---|---|
| requests, numpy, pandas | see `requirements.txt` | plain pip install |
| torch | `2.7.1` (cu126 build) | pinned first, then held with a pip constraints file |
| evo2 | latest (from github) | depends on torch; constrained to not upgrade it |
| flash-attn | `2.8.3.post2` | needs torch already present to build; also constrained |
| appfl | `>=1.12.0`, installed from this repo checkout | last, also constrained |

Without the constraints file, evo2's and appfl's own unpinned `torch`
dependencies silently pull a newer torch mid-install and break the
already-built flash-attn (ABI mismatch at import).

Edit `../client_llnl.yaml` after this: `trainer_path`, `base_data`,
`model_path`, `evo2_dir`, point each at your own paths.

## Creating Batch Script

`slurm_appfl_client.sh` (edit `<username>`, `<llnl-account>`, and the paths
before submitting):

```bash
#!/bin/bash
#SBATCH --job-name=appfl-evo2-site-anl-3
#SBATCH --account=<llnl-account>
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --output=/p/lustre1/<username>/dbforge/wastewater-peft/logs/appfl_client.out
#SBATCH --error=/p/lustre1/<username>/dbforge/wastewater-peft/logs/appfl_client.err

set -euo pipefail

EVO2_BASE=/p/lustre1/<username>/dbforge/evo2_metagenomics
DBFORGE_ROOT=/p/lustre1/<username>/dbforge/wastewater-peft

cd "$DBFORGE_ROOT"
mkdir -p logs

module load conda
conda activate $EVO2_BASE/envs/evo2_fl_env312

unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY || true

export PIP_CACHE_DIR=$EVO2_BASE/.pip_cache
export TMPDIR=$EVO2_BASE/.tmp
export TRITON_CACHE_DIR=$EVO2_BASE/.triton_cache
export HF_HOME=$EVO2_BASE/models/evo2_7b
export HF_HUB_CACHE=$EVO2_BASE/models/evo2_7b/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python "/home/<username>/APPFL/examples/grpc/run_client_evo2.py" \
  --config "/home/<username>/APPFL/examples/resources/configs/evo2_wastewater_peft/client_llnl.yaml"
```

(This file already exists in this directory, edit it in place rather than
retyping it. `launch_client.sh` is the same thing without `#SBATCH` lines,
for an interactive node instead.)

## Submitting

```bash
sbatch slurm_appfl_client.sh
```

You should see:

```
Submitted batch job <job_id>
```

Watch it live:

```bash
tail -f -n 10 /p/lustre1/<username>/dbforge/wastewater-peft/logs/appfl_client.out
```

For an interactive run instead of batch:

```bash
salloc -N 1 -C gpu -q interactive -t 00:30:00 -A <your_llnl_account>
bash launch_client.sh
```
