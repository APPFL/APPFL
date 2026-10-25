#!/bin/bash
# Interactive-job launcher — same env/invocation as slurm_appfl_client.sh,
# just run directly instead of through #SBATCH. Get an interactive node
# first (see README.md), then run this.

set -euo pipefail

EVO2_BASE=/pscratch/sd/<u>/<username>/dbforge/evo2_metagenomics
DBFORGE_ROOT=/pscratch/sd/<u>/<username>/dbforge/wastewater-peft

cd "$DBFORGE_ROOT"
mkdir -p logs

module load conda
conda activate "$EVO2_BASE/envs/evo2_fl_env312"

unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY || true

export PIP_CACHE_DIR="$EVO2_BASE/.pip_cache"
export TMPDIR="$EVO2_BASE/.tmp"
export TRITON_CACHE_DIR="$EVO2_BASE/.triton_cache"
export HF_HOME="$EVO2_BASE/models/evo2_7b"
export HF_HUB_CACHE="$EVO2_BASE/models/evo2_7b/hub"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "=== APPFL Evo2 Client — site_anl_2/lora ==="
echo "Node: $(hostname)"

python "/home/<username>/APPFL/examples/grpc/run_client_evo2.py" \
  --config "/home/<username>/APPFL/examples/resources/configs/evo2_wastewater_peft/client_perlmutter.yaml"

echo "=== Client done ==="
