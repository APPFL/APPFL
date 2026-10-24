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

EVO2_BASE=/pscratch/sd/a/abhijit7/dbforge/evo2_metagenomics
DBFORGE_ROOT=/pscratch/sd/a/abhijit7/dbforge/wastewater-peft

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

python "/global/homes/a/abhijit7/APPFL/examples/grpc/run_client_evo2.py" \
  --config "/global/homes/a/abhijit7/APPFL/examples/resources/configs/evo2_wastewater_peft/client_2.yaml"
