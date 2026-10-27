#!/bin/bash
#SBATCH --job-name=appfl-evo2-site-anl-2
#SBATCH --account=m4994
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --output=/pscratch/sd/a/abhijit7/dbforge/wastewater-peft/logs/appfl_client.out
#SBATCH --error=/pscratch/sd/a/abhijit7/dbforge/wastewater-peft/logs/appfl_client.err

set -euo pipefail

SHARED_ENV=/global/cfs/projectdirs/m4994/common/db-forge-poc/evo2_metagenomics/envs/evo2_fl_env312
SCRATCH_BASE=/pscratch/sd/a/abhijit7/dbforge/evo2_metagenomics
DBFORGE_ROOT=/pscratch/sd/a/abhijit7/dbforge/wastewater-peft

cd "$DBFORGE_ROOT"
mkdir -p logs

module load conda
conda activate "$SHARED_ENV"


unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY || true

export PIP_CACHE_DIR=$SCRATCH_BASE/.pip_cache
export TMPDIR=$SCRATCH_BASE/.tmp
export TRITON_CACHE_DIR=$SCRATCH_BASE/.triton_cache
export HF_HOME=$SCRATCH_BASE/models/evo2_7b
export HF_HUB_CACHE=$SCRATCH_BASE/models/evo2_7b/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python "/global/homes/a/abhijit7/APPFL/examples/grpc/run_client_evo2.py" \
  --config "/global/homes/a/abhijit7/APPFL/examples/resources/configs/evo2_wastewater_peft/client_perlmutter.yaml"
