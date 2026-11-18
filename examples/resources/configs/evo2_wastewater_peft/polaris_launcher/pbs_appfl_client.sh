#!/bin/bash
#PBS -N appfl_evo2_client
#PBS -l select=1:ncpus=32:ngpus=4
#PBS -l walltime=01:00:00
#PBS -l filesystems=grand:home
#PBS -q debug
#PBS -A GeomicVar
#PBS -o /lus/grand/projects/GeomicVar/schunduru/evo2_metagenomics/db-forge-wastewater/logs/appfl_client.out
#PBS -e /lus/grand/projects/GeomicVar/schunduru/evo2_metagenomics/db-forge-wastewater/logs/appfl_client.err

set -euo pipefail

ENV=/lus/grand/projects/GeomicVar/envs/db_forge_evo2
PY=/soft/applications/conda/2025-09-25/mconda3/bin/python3

export CUDA_HOME=/soft/compilers/cudatoolkit/cuda-12.6.0
export PATH="$CUDA_HOME/bin:$PATH"
CUDNN_LIBS="$ENV/lib/python3.12/site-packages/nvidia/cudnn/lib"
export CUDNN_PATH="$ENV/lib/python3.12/site-packages/nvidia/cudnn"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDNN_LIBS:${LD_LIBRARY_PATH:-}"
unset LD_PRELOAD

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

unset http_proxy HTTP_PROXY
export https_proxy=http://140.221.69.93:3128
export HTTPS_PROXY=http://140.221.69.93:3128
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"

unset PYTHONPATH
export PYTHONPATH="$ENV/lib/python3.12/site-packages"
export PYTHONUNBUFFERED=1

echo "=== APPFL Evo2 Client — site_anl_1/lora ==="
echo "Node: $(hostname) | Python: $($PY --version 2>&1)"

"$PY" -u /home/schunduru/APPFL-evo2-wastewater/examples/grpc/run_client_evo2.py \
    --config /home/schunduru/APPFL-evo2-wastewater/examples/resources/configs/evo2_wastewater_peft/client_polaris.yaml

echo "=== Client done ==="
